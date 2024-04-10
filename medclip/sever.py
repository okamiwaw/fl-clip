import copy
import os

import torch

from medclip import constants


class Server:
    def __init__(self,
                 global_model=None,
                 select_model=None,
                 current_round=0,
                 client_ids=None,
                 soft_lambda=0.7
                 ):
        self.global_model = global_model
        self.select_model = select_model
        self.current_round = current_round
        self.weights = {}
        self.client_ids = client_ids
        self.soft_lambda = soft_lambda
        self.person_models = {}
        for client_id in client_ids:
            self.person_models[client_id] = copy.deepcopy(global_model)
        self.client_weights = constants.CLIENTS_WEIGHT

    def receive(self, client_id, global_dict, select_dict, person_model):
        print(f"server receives {client_id}'s model file")
        names = ["global_weights", "select_weights"]
        dicts = [global_dict, select_dict]
        if not self.weights:
            for idx, name in enumerate(names):
                self.weights[name] = copy.deepcopy(dicts[idx])
            self.weights["person_weights"] = {}
            self.weights["person_weights"][client_id] = copy.deepcopy(person_model.state_dict())
        else:
            for idx, name in enumerate(names):
                model_dict = dicts[idx]
                for key in model_dict:
                    if model_dict[key].dtype == torch.float32:
                        self.weights[name][key] += model_dict[key]
            self.weights["person_weights"][client_id] = copy.deepcopy(person_model.state_dict())

    def aggregate(self):
        print("client starts aggregation")
        self.current_round += 1
        weights = self.weights
        global_dict = self.global_model.state_dict()
        select_dict = self.select_model.state_dict()
        dicts = [global_dict, select_dict]
        for idx, model_dict in enumerate(weights.values()):
            for key in model_dict.keys():
                if model_dict[key].dtype == torch.float32:
                    dicts[idx][key] += model_dict[key] * self.client_weights[idx]
        for client_id in self.client_ids:
            person_weight = weights["person_weights"][client_id].copy()
            for key in weights["person_weights"][client_id]:
                if person_weight[key].dtype != torch.float32:
                    continue
                person_weight[key] = person_weight[key] * 0
                for client in self.client_ids:
                    if client == client_id:
                        person_weight[key] += weights["person_weights"][client][key] * self.soft_lambda
                    else:
                        person_weight[key] += weights["person_weights"][client][key] * (1 - self.soft_lambda) / (
                                len(self.client_ids) - 1)
                self.person_models[client_id].load_state_dict(person_weight)

    def save_model(self):
        save_dir = f'outputs/models/{self.current_round}'
        os.makedirs(save_dir, exist_ok=True)
        global_path = os.path.join(save_dir, "global_model.pth")
        torch.save(self.global_model.state_dict(), global_path)
        select_path = os.path.join(save_dir, "select_model.pth")
        torch.save(self.select_model.state_dict(), select_path)
        for client_id in self.client_ids:
            model_path = os.path.join(save_dir, f"person_model_{client_id}.pth")
            torch.save(self.person_models[client_id].state_dict(), model_path)