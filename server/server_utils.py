#server/server_utils.py
import boto3
import os, logging, re
import zipfile
import json
import numpy as np

# FL Server Status Class
class FLServerStatus:
    last_gl_model_v = 0  # Previous Global Model Version
    gl_model_v = 0       # Global model version to be created
    start_by_round = 0   # fit aggregation start
    end_by_round = 0     # fit aggregation end
    round = 0            # round number


# Connect aws session
def aws_session(region_name='ap-northeast-2'):
    return boto3.session.Session(
        aws_access_key_id=os.environ.get('ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('ACCESS_SECRET_KEY'),
        region_name=region_name,
    )


# Global model upload in S3
def upload_model_to_bucket(task_id, global_model_name):
    bucket_name = os.environ.get('BUCKET_NAME')
    if not bucket_name:
        raise RuntimeError("BUCKET_NAME env is not set")

    logging.info(f'bucket_name: {bucket_name}')

    session = aws_session()
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    bucket.upload_file(
        Filename=f'./{global_model_name}',
        Key=f'{task_id}/{global_model_name}',
    )

    logging.info(f'Upload {global_model_name}')


# Download the latest global model stored in s3
def model_download_s3(task_id, model_type, model=None):
    """
    최신 전역모델을 S3에서 찾아 내려받고, model_type에 맞춰 로드해서 반환한다.
    반환: (model, gl_model_name, gl_model_version)
      - 모델이 하나도 없으면 (None, None, 0)
    """
    # 기본 버킷: 환경변수 BUCKET_NAME 우선, 없으면 "global-model"
    bucket_name = os.environ.get('BUCKET_NAME') or "global-model"

    try:
        session = aws_session()
        s3_client = session.client('s3')

        # Prefix는 {task_id}/
        resp = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=f'{task_id}/')

        if 'Contents' not in resp:
            logging.info("No objects found in bucket for this task")
            return None, None, 0

        content_list = resp['Contents']

        # S3에 있는 파일명만 뽑기
        file_list = []
        for content in content_list:
            key = content['Key']
            # 디렉토리 키 스킵
            if key.endswith('/'):
                continue
            file_name = key.split('/', 1)[1] if '/' in key else key
            file_list.append(file_name)

        logging.info(f'model_file_list: {file_list}')

        # File name pattern
        pattern = r"([A-Za-z]+)_gl_model_V(\d+)\.(h5|pth|npz)"
        matching = [f for f in file_list if re.match(pattern, f)]
        if not matching:
            logging.info("No matching global model files in S3")
            return None, None, 0

        # 최신 버전 선택
        latest_gl_model_file = sorted(
            matching,
            key=lambda x: int(re.findall(pattern, x)[0][1]),
            reverse=True
        )[0]

        gl_model_name = re.findall(pattern, latest_gl_model_file)[0][0]
        gl_model_version = int(re.findall(pattern, latest_gl_model_file)[0][1])
        s3_key = f"{task_id}/{latest_gl_model_file}"
        local_path = f"./{latest_gl_model_file}"

        # 다운로드
        s3_client.download_file(bucket_name, s3_key, local_path)
        logging.info(f"Downloaded latest global model: s3://{bucket_name}/{s3_key} -> {local_path}")

        # 로드
        ext = latest_gl_model_file.split(".")[-1].lower()
        if model_type == "Tensorflow" and ext == "h5":
            import tensorflow as tf
            model = tf.keras.models.load_model(local_path)
        elif model_type == "Pytorch" and ext == "pth":
            import torch
            if model is None:
                raise RuntimeError("model_download_s3: For PyTorch, pass an instantiated model to load state_dict into.")
            model.load_state_dict(torch.load(local_path, map_location="cpu"))
        elif model_type == "Huggingface" and ext == "npz":
            # 어댑터 파라미터 등 별도 처리 필요 시 구현
            logging.info("Huggingface npz downloaded (loading must be handled by caller if needed).")
            # 여기서는 파일만 내려주고, 상위에서 불러 쓰도록 둠
        else:
            logging.warning(f"Downloaded file {latest_gl_model_file} does not match model_type={model_type}")
            # 그래도 이름/버전은 리턴
            return model, gl_model_name, gl_model_version

        return model, gl_model_name, gl_model_version

    except Exception as e:
        logging.error(f"model_download_s3 error: {e}")
        return None, None, 0


def model_download_local(model_type, model=None):
    """
    현재 작업 디렉토리에서 최신 전역모델 파일을 찾아 로드.
    반환: (model, gl_model_name, gl_model_version)
    """
    local_list = os.listdir("./")

    pattern = r"([A-Za-z]+)_gl_model_V(\d+)\.(h5|pth)"
    matching_files = [x for x in local_list if re.match(pattern, x)]

    if not matching_files:
        logging.info("No matching model files found locally.")
        return None, None, 0

    latest_gl_model_file = sorted(
        matching_files, key=lambda x: int(re.findall(pattern, x)[0][1]), reverse=True
    )[0]
    gl_model_name = re.findall(pattern, latest_gl_model_file)[0][0]
    gl_model_version = int(re.findall(pattern, latest_gl_model_file)[0][1])

    try:
        if model_type == "Tensorflow" and latest_gl_model_file.endswith(".h5"):
            import tensorflow as tf
            model = tf.keras.models.load_model(latest_gl_model_file)
        elif model_type == "Pytorch" and latest_gl_model_file.endswith(".pth"):
            import torch
            if model is None:
                raise RuntimeError("model_download_local: For PyTorch, pass an instantiated model to load state_dict into.")
            model.load_state_dict(torch.load(latest_gl_model_file, map_location="cpu"))
        else:
            logging.info("No matching loader for file/model_type")
            return None, None, 0
        return model, gl_model_name, gl_model_version
    except Exception as e:
        logging.error(f"Failed to load local model: {e}")
        return None, None, 0


def load_initial_parameters_from_shape(json_path: str):
    """
    parameter_shapes.json 파일을 읽고, 각 shape에 맞는 0으로 초기화된 numpy 파라미터 리스트 반환
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find parameter shape file at: {json_path}")
    
    with open(json_path, "r") as f:
        shape_list = json.load(f)
    
    # numpy 배열 생성 (dtype은 float32가 일반적)
    initial_parameters = [np.zeros(shape, dtype=np.float32) for shape in shape_list]
    return initial_parameters
