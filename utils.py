import os 
import yaml
import pickle
import paddle
import numpy as np
import random
import cv2



def load_config(config_file):
	with open(config_file, 'r') as f:
		config = yaml.safe_load(f)
	return config

def load_model(config, model, optimizer=None, model_type="det"):
    """
    load model from checkpoint or pretrained_model
    """
    global_config = config["Global"]
    checkpoints = global_config.get("checkpoints")
    pretrained_model = global_config.get("pretrained_model")
    best_model_dict = {}
   
    if checkpoints:
        if checkpoints.endswith(".pdparams"):
            checkpoints = checkpoints.replace(".pdparams", "")
        assert os.path.exists(
            checkpoints + ".pdparams"
        ), "The {}.pdparams does not exists!".format(checkpoints)

        # load params from trained model
        params = paddle.load(checkpoints + ".pdparams")
        state_dict = model.state_dict()
        new_state_dict = {}
        for key, value in state_dict.items():
            if key not in params:
                print(
                    "{} not in loaded params {} !".format(key, params.keys())
                )
                continue
            pre_value = params[key]
            if pre_value.dtype == paddle.float16:
                is_float16 = True
            if pre_value.dtype != value.dtype:
                pre_value = pre_value.astype(value.dtype)
            if list(value.shape) == list(pre_value.shape):
                new_state_dict[key] = pre_value
            else:
                print(
                    "The shape of model params {} {} not matched with loaded params shape {} !".format(
                        key, value.shape, pre_value.shape
                    )
                )
        model.set_state_dict(new_state_dict)
        if is_float16:
            print(
                "The parameter type is float16, which is converted to float32 when loading"
            )
        if optimizer is not None:
            if os.path.exists(checkpoints + ".pdopt"):
                optim_dict = paddle.load(checkpoints + ".pdopt")
                optimizer.set_state_dict(optim_dict)
            else:
                print(
                    "{}.pdopt is not exists, params of optimizer is not loaded".format(
                        checkpoints
                    )
                )

        if os.path.exists(checkpoints + ".states"):
            with open(checkpoints + ".states", "rb") as f:
                states_dict = pickle.load(f, encoding="latin1")
            best_model_dict = states_dict.get("best_model_dict", {})
            best_model_dict["acc"] = 0.0
            if "epoch" in states_dict:
                best_model_dict["start_epoch"] = states_dict["epoch"] + 1
        print("resume from {}".format(checkpoints))
        
    elif pretrained_model:
        is_float16 = load_pretrained_params(model, pretrained_model)
    else:
        print("train from scratch")
    best_model_dict["is_float16"] = is_float16
    
    return best_model_dict

def load_pretrained_params(model, path):
    
    if path.endswith(".pdparams"):
        path = path.replace(".pdparams", "")
    assert os.path.exists(
        path + ".pdparams"
    ), "The {}.pdparams does not exists!".format(path)

    params = paddle.load(path + ".pdparams")

    state_dict = model.state_dict()

    new_state_dict = {}
    is_float16 = False

    for k1 in params.keys():
        if k1 not in state_dict.keys():
            print("The pretrained params {} not in model".format(k1))
        else:
            if params[k1].dtype == paddle.float16:
                is_float16 = True
            if params[k1].dtype != state_dict[k1].dtype:
                params[k1] = params[k1].astype(state_dict[k1].dtype)
            if list(state_dict[k1].shape) == list(params[k1].shape):
                new_state_dict[k1] = params[k1]
            else:
                print(
                    "The shape of model params {} {} not matched with loaded params {} {} !".format(
                        k1, state_dict[k1].shape, k1, params[k1].shape
                    )
                )

    model.set_state_dict(new_state_dict)
    if is_float16:
        print(
            "The parameter type is float16, which is converted to float32 when loading"
        )
    print("load pretrain successful from {}".format(path))
    return is_float16

def visualize(image,polys,color = None):
	for poly in polys:
		poly = np.array(poly).reshape(-1, 2)  # Ensure (n, 2) shape
		poly_int = poly.astype(np.int32).reshape(-1, 1, 2)  # For cv2.polylines
		color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) if color is  None else color
		cv2.polylines(image, [poly_int], isClosed=True, color=color, thickness=2)
	return image