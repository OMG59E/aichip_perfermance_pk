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
    target = Column(String(512), comment="目标芯片")
    version = Column(String(512), comment="工具链版本")
    msg = Column(LONGTEXT, comment="编译信息")
    num_core = Column(String(512), comment="配置算力")
    build_time = Column(DateTime, comment="编译时间")
    build_span = Column(String(512), comment="编译耗时")
    latency = Column(Float, default=0, comment="推理时延，单位毫秒")
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
    try:
        basename, ext = os.path.splitext(filename)
        mlir = f"{basename}.mlir"
        if not os.path.exists(mlir) or 1:
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
        if not os.path.exists(cali_table) or 1:
            cmd = f"run_calibration {basename}.mlir --data_list dataset.txt --input_num 1 -o {cali_table}"
            logger.info(cmd)
            run_calibration_cmd = ["run_calibration", f"{basename}.mlir", "--data_list", "dataset.txt", "--input_num", "1", "-o", cali_table]
            res = subprocess.run(run_calibration_cmd, capture_output=True, text=True, check=True, timeout=7200)  # 2小时不结束视为超时
            if res.returncode == 0:
                logger.info(f"run_calibration success:\n{res.stdout}")
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
        if not os.path.exists(bmodel) or 1:
            cmd = f"model_deploy --mlir {basename}.mlir --quantize INT8 --calibration_table {basename}_cali_table --processor {target} --num_core {num_core} --model {bmodel}"
            logger.info(cmd)
            model_deploy_cmd = ["model_deploy", "--mlir", f"{basename}.mlir", "--quantize", "INT8", "--calibration_table", f"{basename}_cali_table", "--processor", target, "--num_core", str(num_core), "--model", bmodel]
            res = subprocess.run(model_deploy_cmd, capture_output=True, text=True, check=True, timeout=7200)  # 2小时不结束视为超时
            if res.returncode == 0:
                logger.info(f"model_deploy success:\n{res.stdout}")
            else:
                msg = f"model_deploy error:\n{res.stderr}"
                logger.error(msg)
                return False, msg
        else:
            logger.info(f"{bmodel} exist")
        return True, "ok"
    except subprocess.CalledProcessError as e:
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"model_deploy exception:\n{e.stderr}")
        return False, str(e.stderr)
    except Exception as e:
        msg = f"model_deploy exception:\n{traceback.format_exc()}"
        logger.error(msg)
        return False, msg
    

def get_latency(filename, md5_code, target, num_core, toolkit_version):
    basename, ext = os.path.splitext(filename)
    ip_addr = "192.168.33.247"
    username = "linaro"
    password = "linaro"
    remote_work_dir = "/home/linaro/"
    key_filename = "/root/.ssh/id_rsa"
    num_iter = 1000
    bmodel = f"{basename}_{target}_{num_core}core_int8_sym_v{toolkit_version}_{md5_code}.bmodel"
    remote_model_path = f"/home/linaro/{bmodel}"
    try:
        with Connection(host=f"{username}@{ip_addr}", connect_kwargs={"key_filename": key_filename}) as c:
            logger.info(f"Upload model: {bmodel} to {remote_work_dir}")
            c.put(bmodel, remote_work_dir)
            result = c.run(f"/opt/sophon/libsophon-current/bin/bmrt_test --calculate_times {num_iter} --bmodel {remote_model_path}", hide=True)
            error = result.stderr.strip()
            if error:
                c.run(f"rm {remote_model_path}", hide=True)
                msg = f"Command Error: {error}, and delete remote model: {remote_model_path}"
                logger.error(msg)
                return 0, msg
            output = result.stdout.strip()
            logger.info(f"Command Output: {output}")
            line = output.splitlines()[-3]
            latency = float(line.strip().split(" ")[-1])
            latency = latency * 1000 / num_iter
            logger.info(f"Exit Code: {result.return_code}")
            c.run(f"rm {remote_model_path}", hide=True)
            logger.info(f"Delete remote model: {remote_model_path}")
            return latency, "ok"
    except Exception as e:
        with Connection(host=f"{username}@{ip_addr}", connect_kwargs={"key_filename": key_filename}) as c:
            c.run(f"rm {remote_model_path}", hide=True)
        msg = f"Run model exception, and delete remote model: {remote_model_path}:\n{traceback.format_exc()}"
        logger.error(msg)
        return 0, msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model build tool")
    parser.add_argument("--models", "-m", type=str, required=True, help="Please specify a onnx model dir")
    parser.add_argument("--target", "-t", type=str, required=True, choices=("bm1688", "bm1684"), help="Please specify a chip target")
    parser.add_argument("--num_core", "-c", type=int, required=False, default=1, choices=(2, 1), help="Please specify a num core, 2 only for bm1688")
    parser.add_argument("--desc", "-d", type=str, required=True, help="Please specify a task desc")
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
    if num_core == 2: 
        assert target == "bm1688", "num_core2 only for bm1688"
    toolkit_version = get_package_version("tpu_mlir")
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
        # 检查该模型已经在数据库
        res = session.query(Models).filter(
            Models.md5_code == md5_code,
            Models.version == toolkit_version,
            Models.target == target,
            Models.num_core == num_core
        ).first()
        if res:
            if res.model_path == filepath:
                logger.info(f"MD5: {md5_code}, Path: {filepath}")
                if bmodel in  ["detr_op11_900x1600_bm1688_2core_int8_sym_583397d68579a08f7cef3d710c5af7e0.bmodel",
                               "detr_r50_8x2_150e_coco_bm1688_2core_int8_sym_7813210d45ccc0f6925ee0d1c79698f8.bmodel",
                               "fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4_P3_demo_post_bm1688_2core_int8_sym_e81155ab869ec43a42906db6975bb906.bmodel",
                               "retinanet_r50_fpn_2x_coco_bm1688_2core_int8_sym_28b1212f7dbc94897003d6bbda117843.bmodel",
                               "ssdlite_mobilenetv2_bm1688_2core_int8_sym_989409e563636c45bb2b5119838ce3fe.bmodel",
                               "yolov7-tiny-nms_bm1688_2core_int8_sym_fd051394ecd2ee18c96296e33caa6573.bmodel",
                               "yolov7-tiny-nms_bm1688_1core_int8_sym_fd051394ecd2ee18c96296e33caa6573.bmodel"]:
                    continue
                if os.path.exists(os.path.join(model_dir, bmodel)) and res.latency == 0:
                    old_dir = os.getcwd()
                    os.chdir(model_dir)
                    # 模型编译生成，但是远程板子推理失败
                    latency, msg = get_latency(filename, md5_code, target, num_core, toolkit_version)
                    if latency != 0:
                        res.msg = "ok"
                        res.latency = latency
                    else:
                        res.msg = msg
                    os.chdir(old_dir)
                    session.add(res)
                    session.commit()
            else:
                # 路径不一致，直接复制
                model = Models()
                model.desc = desc
                model.model_name = filename
                model.model_path = filepath
                model.md5_code = md5_code
                model.version = toolkit_version
                model.target = target
                model.build_time = res.build_time
                model.build_span = res.build_span
                model.compiled_model_path = res.compiled_model_path
                model.compiled_model_md5 = res.compiled_model_md5
                model.num_core = res.num_core
                model.msg = res.msg
                model.latency = res.latency
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
        model.num_core = num_core
        model.target = target
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
        if not os.path.exists(bmodel) or 1:
            t_start = time.time()
            success, msg = build_model(filename, md5_code, target, num_core, toolkit_version)
            span = time.time() - t_start
            model.build_span = str(int(span))
            if not success:
                os.chdir(old_dir)
                model.msg = msg
                session.add(model)
                session.commit()
                continue
        latency, msg = get_latency(filename, md5_code, target, num_core, toolkit_version)
        if latency == 0:
            os.chdir(old_dir)
            model.msg = msg
            session.add(model)
            session.commit()
            continue
        os.chdir(old_dir)
        compiled_model_path = os.path.join(model_dir, bmodel)
        model.compiled_model_path = compiled_model_path
        model.compiled_model_md5 = get_md5_code(compiled_model_path)
        model.latency = latency
        model.msg = "ok"
        session.add(model)
        session.commit()
