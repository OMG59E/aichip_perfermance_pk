import glob
import subprocess
import os, sys
import numpy as np
import json
import shutil
import tarfile
import hashlib
import sqlalchemy
import pymysql
import resource
import onnx
import onnx_graphsurgeon as gs
from natsort import natsorted
from datetime import datetime
from sqlalchemy import Column, Integer, DateTime, String, BOOLEAN, Float, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


HOST = "mysql"
USERNAME = "root"
PASSWORD = "10086"
PORT = 3306
DATABASE = "demodels"

pymysql.install_as_MySQLdb()
engine = sqlalchemy.create_engine(
    "mysql+pymysql://{}:{}@{}:{}".format(USERNAME, PASSWORD, HOST, PORT),
    encoding="utf-8",
    pool_size=100
)  # connect to server
engine.execute("CREATE SCHEMA IF NOT EXISTS `{}`;".format(DATABASE))  # create db
engine.execute("USE `{}`;".format(DATABASE))  # select new db
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base(engine)


class Models(Base):
    __tablename__ = "ax650n"
    __table_args__ = {"extend_existing": True, "mysql_charset": "utf8"}
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(512), index=True, comment="模型名")
    md5_code = Column(String(512), comment="模型md5码")
    model_path = Column(String(512), comment="模型路径")
    version = Column(String(512), comment="爱芯工具链版本")
    status = Column(BOOLEAN, comment="爱芯编译状态")
    msg = Column(String(10240), comment="爱芯编译信息")
    build_time = Column(DateTime, comment="爱芯编译开始时间")
    build_span = Column(String(512), comment="爱芯编译耗时")
    axmodel_path = Column(String(512), comment="爱芯编译后模型路径")
    axmodel_md5 = Column(String(512), comment="爱芯编译后模型MD5")
    latency = Column(Float, default=0, comment="ax650n推理时延，单位毫秒")


# 检查表是否存在
tables = engine.execute("show tables;").fetchall()
tables = [table_name[0] for table_name in tables]
if Models.__tablename__ not in tables:
    Base.metadata.create_all(engine)
    print("create table success")


MAX_MEMORY = 28 * 1024 * 1024 * 1024  # GB

def set_memory_limit(max_mem):
    def limit():
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, hard))
    return limit


def build_ax(config_path, version="2.4"):
    try:
        print("pulsar2 build --config {}".format(config_path))
        # build_cmd = ["docker", "run",
        #              "-v", "/media/xingwg/Data/models:/workspaces/models",
        #              "--rm", "-w", "/workspaces/models", "pulsar2:{}".format(version),
        #              "pulsar2", "build", "--config", config_path
        #              ]
        # res = subprocess.run(build_cmd, preexec_fn=set_memory_limit(MAX_MEMORY),
        #                     capture_output=True, text=True, check=True, timeout=7200)  # 2小时不结束视为超时
        build_cmd = ["pulsar2", "build", "--config", config_path]
        res = subprocess.run(build_cmd, capture_output=True, text=True, check=True, timeout=7200)  # 2小时不结束视为超时
        if res.returncode == 0:
            print("success:\n", res.stdout)
            return True, "ok"
        else:
            print("failed:", res.stderr)
            return False, str(res.stderr)
    except subprocess.TimeoutExpired as e:
        print("exception:", e.stderr)
        return False, str(e.stderr)
    except subprocess.CalledProcessError as e:
        print("exception:", e.stderr)
        return False, str(e.stderr)


def gen_ax_cfg(onnx_path, basename, md5_code, version="2.4"):
    try:
        graph = gs.import_onnx(onnx.load(onnx_path))
        input_names = [node.name for node in graph.inputs]
        input_shapes = [list(node.shape) for node in graph.inputs]
        input_dtypes = [str(node.dtype) for node in graph.inputs]

        npu_mode = "NPU3"
        config = dict()
        config["model_type"] = "ONNX"
        config["npu_mode"] = npu_mode
        config["target_hardware"] = "AX650"
        config["input"] = onnx_path
        config["output_dir"] = "outputs/ax650n/{}/{}_{}/ax650n".format(version, basename, md5_code)
        config["output_name"] = "{}_{}_v{}_{}.axmodel".format(basename, npu_mode, version, md5_code)
        config["quant"] = {"input_configs": []}
        for idx, input_name in enumerate(input_names):
            shape = input_shapes[idx]
            # 对于batch维度是动态的情况，云天可以编译需要改设置
            # if not isinstance(shape[0], int) or shape[0] < 1:
            #     shape[0] = 1
            # if len(shape) >= 2:
            #     for d in range(1, len(shape)):
            #         if not isinstance(shape[d], int) or shape[d] < 1:
            #             print("not support dynamic model:", onnx_path)
            #             return False, "not support dynamic model"
            # 爱芯直接不支持
            for d in range(len(shape)):
                if not isinstance(shape[d], int) or shape[d] < 1:
                    print("not support dynamic model:", onnx_path)
                    return False, "not support dynamic model"
            
            shape_name = "x".join(list(map(str, shape)))
            dtype = input_dtypes[idx]
            if dtype not in ["int64", "uint64", "int32", "uint32", "int8", "uint8", "int16", "uint16", "float32", "float64", "float16", "bool"]:
                print("dtype error:", dtype)
                return False, "not support input dtype: {}".format(dtype)
            
            basename = "{}_{}".format(shape_name, dtype)
            bin_path = os.path.join("data", basename)
            if not os.path.exists(bin_path):
                os.makedirs(bin_path)
            bin_fullpath = os.path.join(bin_path, "{}_{}.bin".format(shape_name, dtype))
            tar_filepath = os.path.join("data", "{}.tar".format(basename))
            if not os.path.exists(tar_filepath):
                if dtype in ["int64", "uint64", "int32", "uint32", "int8", "uint8", "int16", "uint16", "bool"]:
                    data = np.random.randint(0, 255, size=shape).astype(dtype=dtype)
                else:
                    data = np.random.uniform(0, 1, size=shape).astype(dtype=dtype)
                data.tofile(bin_fullpath)
                with tarfile.open(tar_filepath, "w") as f:
                    f.add(bin_path)
            input_config = dict()
            input_config["tensor_name"] = input_name
            input_config["calibration_dataset"] = tar_filepath
            input_config["calibration_format"] = "Binary"
            input_config["calibration_size"] = 1
            config["quant"]["input_configs"].append(input_config)
        config["quant"]["calibration_method"] = "MinMax"
        config["quant"]["precision_analysis"] = False
    
        json_str = json.dumps(config, indent=4, sort_keys=False)
        with open(config_path, "w") as f:
            f.write(json_str)
        print("save config file to:", config_path)
        return True, "ok"
    except Exception as e:
        print("Failed to gen config:", e)
        return False, str(e)


def get_md5_code(filepath):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            md5.update(chunk)
    return md5.hexdigest()


if __name__ == "__main__":
    models_dir = sys.argv[1]
    target = "ax650n"
    toolkit_version = "2.4"
    config_dir = "configs/{}/{}".format(target, toolkit_version)
    complied_models = "models/{}/{}".format(target, toolkit_version)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    if not os.path.exists(complied_models):
        os.makedirs(complied_models)
    onnx_files = glob.glob(os.path.join(models_dir, "**/*.onnx"), recursive=True)
    sorted_file_list = natsorted(onnx_files, key=os.path.basename)
    for idx, filepath in enumerate(sorted_file_list):
        if not os.path.exists(filepath):
            print("Not found:", filepath)
            continue
        
        md5_code = get_md5_code(filepath)
        filename = os.path.basename(filepath)
        basename, ext = os.path.splitext(filename)
        
        # 检查该模型是不是已经入库
        res = session.query(Models).filter(
            Models.md5_code == md5_code,
            Models.version == toolkit_version
        ).first()
        if res:
            if res.model_name == filename:
                print("{} has exist, md5: {}".format(filename, md5_code))
            else:
                # 名字不一样，但是md5相同的模型，直接复制
                model = Models()
                model.model_name = filename
                model.model_path = filepath
                model.md5_code = md5_code
                model.version = toolkit_version
                model.build_time = res.build_time
                model.build_span = res.build_span
                model.axmodel_path = res.axmodel_path
                model.axmodel_md5 = res.axmodel_md5
                model.status = res.status
                model.msg = res.msg
                session.add(model)
                session.commit()
            continue
        
        # 新模型
        model = Models()
        model.model_name = filename
        model.model_path = filepath
        model.md5_code = md5_code
        model.version = toolkit_version
        build_time = datetime.now()
        model.build_time = build_time
          
        config_path = os.path.join(config_dir, "{}_{}.json".format(basename, md5_code))
        success, msg = gen_ax_cfg(filepath, basename, md5_code, toolkit_version)
        if not success:
            model.msg = msg
            session.add(model)
            session.commit()
            continue
        
        with open(config_path, "r") as f:
            config = json.load(f)
        axmodel_path = os.path.join(config["output_dir"], config["output_name"])
        success, msg = build_ax(config_path)
        if len(msg) > 10240:
            msg = msg[:10240]
        model.build_span = str(datetime.now() - build_time)
        if not success:
            print("Failed to build:", filepath)
            model.status = False
            model.msg = msg
            session.add(model)
            session.commit()
            continue
        model.axmodel_path = axmodel_path
        model.axmodel_md5 = get_md5_code(axmodel_path)
        model.status = True
        model.msg = "ok"
        session.add(model)
        session.commit()
        shutil.copy(axmodel_path, complied_models)

