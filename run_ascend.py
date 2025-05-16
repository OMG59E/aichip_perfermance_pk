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
from onnx import helper
from onnxsim import simplify
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
    __tablename__ = "ascend"
    __table_args__ = {"extend_existing": True, "mysql_charset": "utf8"}
    id = Column(Integer, primary_key=True, autoincrement=True)
    desc = Column(String(512), comment="任务描述")
    model_name = Column(String(512), index=True, comment="模型名")
    md5_code = Column(String(512), comment="md5")
    model_path = Column(String(512), comment="模型路径")
    batch_size = Column(String(16), comment="模型路径")
    target = Column(String(512), comment="目标芯片")
    version = Column(String(512), comment="工具链版本")
    msg = Column(LONGTEXT, comment="编译信息")
    build_time = Column(DateTime, comment="编译时间")
    span = Column(String(512), comment="编译耗时")
    latency = Column(Float, default=0, comment="单位毫秒")
    ddr_r = Column(Float, default=0, comment="DDR读带宽 单位MB/s")
    ddr_w = Column(Float, default=0, comment="DDR写带宽 单位MB/s")
    compiled_model_path = Column(String(512), comment="编译后模型路径")
    compiled_model_md5 = Column(String(512), comment="编译后模型MD5")
    

def get_md5_code(filepath):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            md5.update(chunk)
    return md5.hexdigest()


def copy_to(filepath, model_dir, opset_version=12):
    filename = os.path.basename(filepath)
    model = onnx.load(filepath)
    # 创建一个新的模型图
    new_graph = helper.make_graph(
        nodes=model.graph.node,                # 保留所有节点
        name="by intellif",
        inputs=model.graph.input,              # 保留输入
        outputs=model.graph.output,            # 保留输出
        initializer=model.graph.initializer,   # 保留初始化器
        value_info=None
    )
    while len(model.opset_import) > 0:
        model.opset_import.pop()
    model.opset_import.append(helper.make_opsetid(domain="", version=opset_version))
    new_model = helper.make_model(new_graph, producer_name=model.producer_name, opset_imports=model.opset_import)
    try:
        model_sim, success = simplify(new_model)
    except Exception as e:
        success = False
    work_model_path = os.path.join(model_dir, filename)
    if not success:
        logger.error(f"Failed to process: {filepath}")
        onnx.save(new_model, work_model_path)
    else:
        logger.info(f"Save model: {work_model_path}")
        onnx.save(model_sim, work_model_path)


def build_model(filename, md5_code, target, toolkit_version):
    build_info = {"span": "0"}
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
        compiled_model_path = f"{basename}.om"
        if not os.path.exists(compiled_model_path) or re_build:
            t0 = time.time()
            cmd = f"atc --model {filename} --output {basename} --framework 5 --soc_version {target}"
            logger.info(cmd)
            build_cmd = ["atc", "--model", filename, "--output", basename, "--framework", "5", "--soc_version", target]
            res = subprocess.run(build_cmd, capture_output=True, text=True, check=True, timeout=7200)  # 2小时不结束视为超时
            if res.returncode == 0:
                logger.info(f"model convert success:\n{res.stdout}")
                build_info["span"] = f"{time.time() - t0:.3f}"
            else:
                msg = f"model convert error:\n{res.stderr}"
                logger.error(msg)
                return False, msg
        else:
            logger.info(f"{compiled_model_path} exist")
    except subprocess.CalledProcessError as e:
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"model convert exception:\n{e.stderr}")
        return False, str(e.stderr)
    except subprocess.TimeoutExpired as e:
        msg = f"model convert exception:\n{e.stderr}"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"model convert exception:\n{traceback.format_exc()}"
        logger.error(msg)
        return False, msg

    with open(build_info_file, "w") as f:
        json.dump(build_info, f, sort_keys=False, indent=4)
    return True, "ok"


def get_latency(filename, md5_code, target, toolkit_version):
    basename, ext = os.path.splitext(filename)
    ip_addr = "192.168.33.247"
    username = "root"
    password = " "
    remote_work_dir = "/root/"
    key_filename = "/root/.ssh/id_rsa"
    num_iter = 100
    compiled_model_name = f"{basename}.om"
    remote_model_path = f"/root/{compiled_model_name}"
    try:
        with Connection(host=f"{username}@{ip_addr}", port=8122, connect_kwargs={"key_filename": key_filename}) as c:
            logger.info(f"Upload model: {compiled_model_name} to {remote_work_dir}")
            c.put(compiled_model_name, remote_work_dir)
            result = c.run(f"./acl_run_model.sh {remote_model_path} {num_iter}", hide=True)
            error = result.stderr.strip()
            if error:
                c.run(f"rm {remote_model_path}", hide=True)
                msg = f"Command Error: {error}, and delete remote model: {remote_model_path}"
                logger.error(msg)
                return 0, msg
            output = result.stdout.strip()
            logger.info(f"Command Output: {output}")
            lines = output.splitlines()
            last_line = lines[-1].strip()
            latency = float(last_line.split(" ")[1][:-2])
            logger.info(f"Exit Code: {result.return_code}")
            c.run(f"rm {remote_model_path}", hide=True)
            logger.info(f"Delete remote model: {remote_model_path}")
            return latency, "ok"
    except Exception as e:
        with Connection(host=f"{username}@{ip_addr}", port=8122, connect_kwargs={"key_filename": key_filename}) as c:
            c.run(f"rm {remote_model_path}", hide=True)
        msg = f"Run model exception, and delete remote model: {remote_model_path}:\n{traceback.format_exc()}"
        logger.error(msg)
        return 0, msg
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model build tool")
    parser.add_argument("type", type=str, choices=("build", "copy"), help="Please specify a operator")
    parser.add_argument("--models", "-m", type=str, required=True, help="Please specify a onnx model dir")
    parser.add_argument("--target", "-t", type=str, required=True, choices=("Ascend310", "Ascend310B1", "Ascend310P3"), help="Please specify a chip target")
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
    run_type = args.type
    toolkit_version = "8.0.0.alpha002"
    
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
        model_dir = f"{work_dir}/{basename}_{md5_code}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        work_model_path = os.path.join(model_dir, filename)
        if not os.path.exists(work_model_path):
            copy_to(filepath, model_dir)

        # 检查该模型已经在数据库
        res = session.query(Models).filter(
            Models.md5_code == md5_code,
            Models.version == toolkit_version,
            Models.target == target,
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
                model.build_time = res.build_time
                model.span = res.span
                model.compiled_model_path = res.compiled_model_path
                model.compiled_model_md5 = res.compiled_model_md5
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
        model.target = target

        old_dir = os.getcwd()
        os.chdir(model_dir)
        logger.info(f"Change work dir to {model_dir}")
        
        success, msg = build_model(filename, md5_code, target, toolkit_version)
        if not success:
            os.chdir(old_dir)
            model.msg = msg
            session.add(model)
            session.commit()
            continue
        if not os.path.exists(f"{basename}.om"):
            os.chdir(old_dir)
            model.msg = "build failed"
            session.add(model)
            session.commit()
            continue 
        
        # 编译成功后，更新编译时间
        build_info_file = "build_info.json"
        if os.path.exists(build_info_file):
            with open(build_info_file, "r") as f:
                build_info = json.load(f)
            model.span = build_info["span"]
        
        # 上传板子获取时延
        latency, msg = get_latency(filename, md5_code, target, toolkit_version)
        if msg == "ok":
            model.latency = latency

        os.chdir(old_dir)
        compiled_model_path = os.path.join(model_dir, f"{basename}.om")
        model.compiled_model_path = compiled_model_path
        model.compiled_model_md5 = get_md5_code(compiled_model_path)
        model.msg = "ok"
        session.add(model)
        session.commit()
