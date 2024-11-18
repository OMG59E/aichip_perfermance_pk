import glob
import subprocess
import os
import sys
import json
import shutil
import traceback
import tarfile
import hashlib
import yaml
import time
import sqlalchemy
import pymysql
import resource
import argparse
import onnx
import onnx_graphsurgeon as gs
import numpy as np
from loguru import logger
from fabric import Connection
from natsort import natsorted
from datetime import datetime
from sqlalchemy.dialects.mysql import LONGTEXT
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
    f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}",
    encoding="utf-8",
    pool_size=100
)  # connect to server
engine.execute(f"CREATE SCHEMA IF NOT EXISTS `{DATABASE}`;")  # create db
engine.execute(f"USE `{DATABASE}`;")  # select new db
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base(engine)
    
class Models(Base):
    __tablename__ = "sophgo"
    __table_args__ = {"extend_existing": True, "mysql_charset": "utf8"}
    id = Column(Integer, primary_key=True, autoincrement=True)
    desc = Column(String(512), comment="任务描述")
    model_name = Column(String(512), index=True, comment="模型名")
    md5_code = Column(String(512), comment="md5")
    model_path = Column(String(512), comment="模型路径")
    batch_size = Column(String(16), comment="模型路径")
    target = Column(String(512), comment="目标芯片")
    num_core = Column(String(1), comment="目标芯片")
    version = Column(String(512), comment="工具链版本")
    msg = Column(LONGTEXT, comment="编译信息")
    build_time = Column(DateTime, comment="编译时间")
    transform_span = Column(String(512), comment="转换耗时")
    calibration_span = Column(String(512), comment="量化耗时")
    deploy_span = Column(String(512), comment="编译耗时")
    latency_1c1t1c = Column(Float, default=0, comment="1core模型, 1个线程,   1个线程独立占用1core, 单位毫秒")
    latency_1c2t1c = Column(Float, default=0, comment="1core模型, 2个线程, 每1个线程共同占用1core, 单位毫秒")
    latency_1c2t2c = Column(Float, default=0, comment="1core模型, 2个线程, 每1个线程分别占用1core, 单位毫秒")
    latency_1c4t2c = Column(Float, default=0, comment="1core模型, 4个线程, 每2个线程共同占用1core, 单位毫秒")
    latency_2c1t2c = Column(Float, default=0, comment="2core模型, 1个线程,   1个线程独立占用2core, 单位毫秒")
    latency_2c2t2c = Column(Float, default=0, comment="2core模型, 2个线程,   1个线程共同占用2core, 单位毫秒")
    get_output_time_1c1t1c = Column(Float, default=0, comment="")
    get_output_time_1c2t1c = Column(Float, default=0, comment="")
    get_output_time_1c2t2c = Column(Float, default=0, comment="")
    get_output_time_1c4t2c = Column(Float, default=0, comment="")
    get_output_time_2c1t2c = Column(Float, default=0, comment="")
    get_output_time_2c2t2c = Column(Float, default=0, comment="")
    compiled_model_path = Column(String(512), comment="编译后模型路径")
    compiled_model_md5 = Column(String(512), comment="编译后模型MD5")
    

def set_memory_limit():
    max_memory = 32 * 1024 * 1024 * 1024 # 32GB
    resource.setrlimit(resource.RLIMIT_AS, (max_memory, resource.RLIM_INFINITY))
    

def get_md5_code(filepath):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            md5.update(chunk)
    return md5.hexdigest()


def get_package_version(package_name):
    import pkg_resources
    try:
        version = pkg_resources.get_distribution(package_name).version
        return version
    except pkg_resources.DistributionNotFound:
        return f"Package {package_name} is not installed."
    
    
def gen_data(filename):
    try:
        # 检查onnx是否仅为batch维度动态
        graph = gs.import_onnx(onnx.load(filename))
        input_names = [node.name for node in graph.inputs]
        input_shapes = [list(node.shape) for node in graph.inputs]
        input_dtypes = [str(node.dtype) for node in graph.inputs]
        npy_list = list()
        for idx, input_name in enumerate(input_names):
            shape = input_shapes[idx]
            if not isinstance(shape[0], int) or shape[0] < 1:
                shape[0] = 1
            if len(shape) >= 2:
                for d in range(1, len(shape)):
                    if not isinstance(shape[d], int) or shape[d] < 1:
                        msg = f"not support dynamic shape: {shape}"
                        logger.error(msg)
                        return False, msg

            shape_name = "x".join(list(map(str, shape)))
            dtype = input_dtypes[idx]
            if dtype not in ["int64", "uint64", "int32", "uint32", "int8", "uint8", "int16", "uint16", "float32", "float64", "float16", "bool"]:
                msg = f"not support input dtype: {dtype}"
                logger.error(msg)
                return False, msg

            npy_name = f"{input_name}_{shape_name}_{dtype}.npy"
            if not os.path.exists(npy_name):
                if dtype in ["int64", "uint64", "int32", "uint32", "int8", "uint8", "int16", "uint16", "bool"]:
                    data = np.random.randint(0, 255, size=shape).astype(dtype=dtype)
                else:
                    data = np.random.uniform(0, 1, size=shape).astype(dtype=dtype)
                np.save(npy_name, data)
            npy_list.append(npy_name)
        with open("dataset.txt", "w") as f:
            f.write(",".join(npy_list))
        return True, "ok"
    except Exception as e:
        msg = f"Generate random data exception:\n{e}"
        logger.error(msg)
        return False, msg


def build_model(filename, md5_code, target, num_core, toolkit_version):
    build_info = {"transform": "0", "calibration": "0", "deploy": "0"}
    build_info_file = "build_info.json"
    re_build = False 
    if os.path.exists(build_info_file):
        with open(build_info_file, "r") as f:
            build_info_old = json.load(f)
            build_info.update(build_info_old)
    else:
        re_build = True
        
    try:
        basename, ext = os.path.splitext(filename)
        mlir = f"{basename}.mlir"
        if not os.path.exists(mlir) or re_build:
            t0 = time.time()
            # get input shapes
            with open("dataset.txt", "r") as f:
                lines = f.readlines()
            npy_list = lines[0].strip().split(",")
            input_shapes = []
            for npy_name in npy_list:
                shape = list(np.load(npy_name).shape)
                input_shapes.append(shape)
            cmd = f"model_transform --model_name {basename} --input_shapes {input_shapes} --model_def {filename} --mlir {mlir}"
            logger.info(cmd)
            model_transform_cmd = ["model_transform", "--model_name", basename, "--input_shapes", f"{input_shapes}", "--model_def", filename, "--mlir", mlir]
            res = subprocess.run(model_transform_cmd, capture_output=True, text=True, check=True, timeout=7200)  # 2小时不结束视为超时
            if res.returncode == 0:
                logger.info(f"model_transform success:\n{res.stdout}")
                build_info["transform"] = f"{time.time() - t0:.3f}"
            else:
                msg = f"model_transform error:\n{res.stderr}"
                logger.error(msg)
                return False, msg
        else:
            logger.info(f"{mlir} exist")
    except subprocess.CalledProcessError as e:
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"model_transform exception:\n{e.stderr}")
        return False, str(e.stderr)
    except subprocess.TimeoutExpired as e:
        msg = f"model_transform exception:\n{e.stderr}"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"model_transform exception:\n{traceback.format_exc()}"
        logger.error(msg)
        return False, msg
    
    try:
        cali_table = f"{basename}_cali_table"
        if not os.path.exists(cali_table) or re_build:
            t0 = time.time()
            cmd = f"run_calibration {basename}.mlir --data_list dataset.txt --input_num 1 -o {cali_table}"
            logger.info(cmd)
            run_calibration_cmd = ["run_calibration", f"{basename}.mlir", "--data_list", "dataset.txt", "--input_num", "1", "-o", cali_table]
            res = subprocess.run(run_calibration_cmd, capture_output=True, text=True, check=True, timeout=7200)  # 2小时不结束视为超时
            if res.returncode == 0:
                logger.info(f"run_calibration success:\n{res.stdout}")
                build_info["calibration"] = f"{time.time() - t0:.3f}"
            else:
                msg = f"run_calibration error:\n{res.stderr}"
                logger.error(msg)
                return False, msg
        else:
            logger.info(f"{cali_table} exist")
    except subprocess.CalledProcessError as e:
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"run_calibration exception:\n{e.stderr}")
        return False, str(e.stderr)
    except subprocess.TimeoutExpired as e:
        msg = f"run_calibration exception:\n{e.stderr}"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"run_calibration exception:\n{traceback.format_exc()}"
        logger.error(msg)
        return False, msg
    
    try:
        bmodel = f"{basename}_{target}_{num_core}core_int8_sym_v{toolkit_version}_{md5_code}.bmodel"
        if not os.path.exists(bmodel) or re_build:
            t0 = time.time()
            cmd = f"model_deploy --mlir {basename}.mlir --quantize INT8 --calibration_table {basename}_cali_table --processor {target} --num_core {num_core} --model {bmodel}"
            logger.info(cmd)
            model_deploy_cmd = ["model_deploy", "--mlir", f"{basename}.mlir", "--quantize", "INT8", "--calibration_table", f"{basename}_cali_table", "--processor", target, "--num_core", str(num_core), "--model", bmodel]
            res = subprocess.run(model_deploy_cmd, capture_output=True, text=True, check=True, timeout=7200)  # 2小时不结束视为超时
            if res.returncode == 0:
                logger.info(f"model_deploy success:\n{res.stdout}")
                build_info["deploy"] = f"{time.time() - t0:.3f}"
            else:
                msg = f"model_deploy error:\n{res.stderr}"
                logger.error(msg)
                return False, msg
        else:
            logger.info(f"{bmodel} exist")
    except subprocess.CalledProcessError as e:
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"model_deploy exception:\n{e.stderr}")
        return False, str(e.stderr)
    except Exception as e:
        msg = f"model_deploy exception:\n{traceback.format_exc()}"
        logger.error(msg)
        return False, msg

    with open(build_info_file, "w") as f:
        json.dump(build_info, f, sort_keys=False, indent=4)
    return True, "ok"
    

def get_latency(filename, md5_code, target, num_core, mode, toolkit_version):
    basename, ext = os.path.splitext(filename)
    ip_addr = "192.168.33.247"
    username = "linaro"
    password = "linaro"
    remote_work_dir = "/home/linaro/"
    key_filename = "/root/.ssh/id_rsa"
    num_iter = 100
    bmodel = f"{basename}_{target}_{num_core}core_int8_sym_v{toolkit_version}_{md5_code}.bmodel"
    remote_model_path = f"/home/linaro/{bmodel}"
    try:
        with Connection(host=f"{username}@{ip_addr}", connect_kwargs={"key_filename": key_filename}) as c:
            logger.info(f"Upload model: {bmodel} to {remote_work_dir}")
            c.put(bmodel, remote_work_dir)
            # result = c.run(f"/opt/sophon/libsophon-current/bin/bmrt_test --calculate_times {num_iter} --bmodel {remote_model_path}", hide=True)
            result = c.run(f"./bm_run_model2 {remote_model_path} {mode} {num_iter}", hide=True)
            error = result.stderr.strip()
            if error:
                c.run(f"rm {remote_model_path}", hide=True)
                msg = f"Command Error: {error}, and delete remote model: {remote_model_path}"
                logger.error(msg)
                return 0, 0, 0, msg
            output = result.stdout.strip()
            logger.info(f"Command Output: {output}")
            lines = output.splitlines()
            last_line = lines[-1].strip()
            latency = float(last_line.split(" ")[1][:-2])
            throughput = float(last_line.split(" ")[3])
            get_output_time = float(last_line.split(" ")[5][:-2])
            logger.info(f"Exit Code: {result.return_code}")
            c.run(f"rm {remote_model_path}", hide=True)
            logger.info(f"Delete remote model: {remote_model_path}")
            return latency, throughput, get_output_time, "ok"
    except Exception as e:
        with Connection(host=f"{username}@{ip_addr}", connect_kwargs={"key_filename": key_filename}) as c:
            c.run(f"rm {remote_model_path}", hide=True)
        msg = f"Run model exception, and delete remote model: {remote_model_path}:\n{traceback.format_exc()}"
        logger.error(msg)
        return 0, msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model build tool")
    parser.add_argument("type", type=str, choices=("build", "copy"), help="Please specify a operator")
    parser.add_argument("--models", "-m", type=str, required=True, help="Please specify a onnx model dir")
    parser.add_argument("--target", "-t", type=str, required=True, choices=("bm1688", "bm1684"), help="Please specify a chip target")
    parser.add_argument("--num_core", "-c", type=int, required=False, default=1, choices=(2, 1), help="Please specify a num core, 2 only for bm1688")
    parser.add_argument("--desc", "-d", type=str, required=False if "copy" in sys.argv else True, help="Please specify a task desc")
    parser.add_argument("--output", "-o", type=str, required=True if "copy" in sys.argv else False, help="Please specify a output path")
    args = parser.parse_args()
    
    # 检查表是否存在
    tables = engine.execute("show tables;").fetchall()
    tables = [table_name[0] for table_name in tables]
    if Models.__tablename__ not in tables:
        Base.metadata.create_all(engine)
        logger.info("create table success")

    models_dir = args.models
    target = args.target
    desc = args.desc
    num_core = args.num_core
    run_type = args.type
    if num_core == 2: 
        assert target == "bm1688", "num_core2 only for bm1688"
    toolkit_version = str(get_package_version("tpu_mlir"))
    logger.info(f"Toolkit Version: {toolkit_version}")
    work_dir = f"outputs/{target}/{toolkit_version}"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    onnx_files = glob.glob(os.path.join(models_dir, "**/*.onnx"), recursive=True)
    sorted_file_list = natsorted(onnx_files, key=os.path.basename)
    for idx, filepath in enumerate(sorted_file_list):
        if not os.path.exists(filepath):
            logger.warning(f"Not found: {filepath}")
            continue
        
        md5_code = get_md5_code(filepath)
        filename = os.path.basename(filepath)
        basename, ext = os.path.splitext(filename)
        model_dir = f"{work_dir}/{basename}_{md5_code}/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(os.path.join(model_dir, filename)):
            shutil.copy(filepath, model_dir)
        
        bmodel = f"{basename}_{target}_{num_core}core_int8_sym_v{toolkit_version}_{md5_code}.bmodel"
        
        if run_type == "copy":
            output_dir = args.output
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            bmodel_path = os.path.join(model_dir, bmodel)
            output_path = os.path.join(output_dir, bmodel)
            if os.path.exists(bmodel_path) and not os.path.exists(output_path):
                shutil.copy(bmodel_path, output_path)
                logger.info(f"copy: {bmodel_path} -> {output_path}")
            continue
        
        # 检查该模型已经在数据库
        res = session.query(Models).filter(
            Models.md5_code == md5_code,
            Models.version == toolkit_version,
            Models.target == target,
            Models.num_core == num_core,
        ).first()
        if res:
            if res.model_path == filepath:
                logger.info(f"MD5: {md5_code}, Path: {filepath}")
            else:
                # 路径不一致，直接复制
                model = Models()
                model.desc = desc
                model.model_name = filename
                model.model_path = filepath
                model.md5_code = md5_code
                model.version = toolkit_version
                model.target = target
                model.num_core = num_core
                model.build_time = res.build_time
                model.transform_span = res.transform_span
                model.calibration_span = res.calibration_span
                model.deploy_span = res.deploy_1core_span
                model.compiled_model_path = res.compiled_model_path
                model.compiled_model_md5 = res.compiled_model_md5
                model.msg = res.msg
                model.latency_1c1t1c = res.latency_1c1t1c 
                model.latency_1c2t1c = res.latency_1c2t1c
                model.latency_1c2t2c = res.latency_1c2t2c
                model.latency_1c4t2c = res.latency_1c4t2c
                model.latency_2c1t2c = res.latency_2c1t2c
                model.latency_2c2t2c = res.latency_2c2t2c
                model.get_output_time_1c1t1c = res.get_output_time_1c1t1c 
                model.get_output_time_1c2t1c = res.get_output_time_1c2t1c
                model.get_output_time_1c2t2c = res.get_output_time_1c2t2c
                model.get_output_time_1c4t2c = res.get_output_time_1c4t2c
                model.get_output_time_2c1t2c = res.get_output_time_2c1t2c
                model.get_output_time_2c2t2c = res.get_output_time_2c2t2c
                session.add(model)
                session.commit()
            continue
    
        model = Models()
        model.desc = desc
        model.model_name = filename
        model.model_path = filepath
        model.md5_code = md5_code
        model.version = toolkit_version
        build_time = datetime.now()
        model.build_time = build_time
        model.target = target
        model.num_core = num_core
        old_dir = os.getcwd()
        os.chdir(model_dir)
        logger.info(f"Change work dir to {model_dir}")
        logger.info("Generate random data for quantization")
        success, msg = gen_data(filename) 
        if not success:
            os.chdir(old_dir)
            model.msg = msg
            session.add(model)
            session.commit()
            continue

        success, msg = build_model(filename, md5_code, target, num_core, toolkit_version)
        if not success:
            os.chdir(old_dir)
            model.msg = msg
            session.add(model)
            session.commit()
            continue
        
        # 编译成功后，更新编译时间
        build_info_file = "build_info.json"
        if os.path.exists(build_info_file):
            with open(build_info_file, "r") as f:
                build_info = json.load(f)
            model.transform_span = build_info["transform"]
            model.calibration_span = build_info["calibration"]
            model.deploy_span = build_info["deploy"]
        
        # 上传板子获取时延
        if num_core == 1:
            # 1c1t1c
            latency, throughput, get_output_time, msg = get_latency(filename, md5_code, target, 1, 0, toolkit_version)
            if msg == "ok":
                model.latency_1c1t1c = latency
                model.get_output_time_1c1t1c = get_output_time
            # 1c2t1c
            latency, throughput, get_output_time, msg = get_latency(filename, md5_code, target, 1, 1, toolkit_version)
            if msg == "ok":
                model.latency_1c2t1c = latency
                model.get_output_time_1c2t1c = get_output_time
            # 1c2t2c
            latency, throughput, get_output_time, msg = get_latency(filename, md5_code, target, 1, 2, toolkit_version)
            if msg == "ok":
                model.latency_1c2t2c = latency
                model.get_output_time_1c2t2c = get_output_time
            # 1c4t2c
            latency, throughput, get_output_time, msg = get_latency(filename, md5_code, target, 1, 3, toolkit_version)
            if msg == "ok":
                model.latency_1c4t2c = latency
                model.get_output_time_1c4t2c = get_output_time
        else:
            # 2c1t2c
            latency, throughput, get_output_time, msg = get_latency(filename, md5_code, target, 2, 4, toolkit_version)
            if msg == "ok":
                model.latency_2c1t2c = latency
                model.get_output_time_2c1t2c = get_output_time
            # 2c2t2c
            latency, throughput, get_output_time, msg = get_latency(filename, md5_code, target, 2, 5, toolkit_version)
            if msg == "ok":
                model.latency_2c2t2c = latency
                model.get_output_time_2c2t2c = get_output_time
        os.chdir(old_dir)
        compiled_model_path = os.path.join(model_dir, bmodel)
        model.compiled_model_path = compiled_model_path
        model.compiled_model_md5 = get_md5_code(compiled_model_path)
        model.msg = "ok"
        session.add(model)
        session.commit()
