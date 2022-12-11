from scripts.utils import rotation_discrete, model_frozen, train_and_val_model, experiment_store, final_metrics_predict
import torch
from torchvision import transforms
import torchmetrics

#loading data
train_set=torch.load("data/pytorch/train_set.pt")
test_set=torch.load("data/pytorch/test_set.pt")
val_set=torch.load("data/pytorch/val_set.pt")


# experiment 1 (baseline)
model1=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model1.fc=torch.nn.Linear(512, 1)
model1.to('cuda')

optimizer1=torch.optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)


experiment1={"exp_number": 1, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp1_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment1["best_rmse"]=train_and_val_model(model1, optimizer1, torch.nn.MSELoss() , experiment1["data_aug"], 
                                            train_set, experiment1["train_batch_size"], val_set, experiment1["val_batch_size"] ,
                                            experiment1["num_epochs"], experiment1["params_path"],experiment1["train_rmse"], experiment1["val_rmse"])

#storing results of the experiment
experiment_store(experiment1,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")


#experiment 2 (base + data augmentation)
model2=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model2.fc=torch.nn.Linear(512, 1)
model2.to('cuda')

optimizer2=torch.optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)


experiment2={"exp_number": 2, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.4),
                                  transforms.RandomHorizontalFlip(p=0.4),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.4,0.2,0.2,0.2]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp2_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment2["best_rmse"]=train_and_val_model(model2, optimizer2, torch.nn.MSELoss() , experiment2["data_aug"], 
                                            train_set, experiment2["train_batch_size"], val_set, experiment2["val_batch_size"] ,
                                            experiment2["num_epochs"], experiment2["params_path"],experiment2["train_rmse"], experiment2["val_rmse"])

#storing results of the experiment
experiment_store(experiment2,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 3 (1 unfrozen + data aug)
model3=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model3.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model3=model_frozen(model3,n_unfreeze=1).to('cuda')

optimizer3=torch.optim.SGD(model3.parameters(), lr=0.001, momentum=0.9)


experiment3={"exp_number": 3, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.4),
                                  transforms.RandomHorizontalFlip(p=0.4),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.4,0.2,0.2,0.2]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp3_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 1 }


experiment3["best_rmse"]=train_and_val_model(model3, optimizer3, torch.nn.MSELoss() , experiment3["data_aug"], 
                                            train_set, experiment3["train_batch_size"], val_set, experiment3["val_batch_size"] ,
                                            experiment3["num_epochs"], experiment3["params_path"],experiment3["train_rmse"], experiment3["val_rmse"])

#storing results of the experiment
experiment_store(experiment3,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 4 (2 unfrozen + data aug)
model4=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model4.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model4=model_frozen(model4,n_unfreeze=2).to('cuda')

optimizer4=torch.optim.SGD(model4.parameters(), lr=0.001, momentum=0.9)


experiment4={"exp_number": 4, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.4),
                                  transforms.RandomHorizontalFlip(p=0.4),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.4,0.2,0.2,0.2]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp4_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 2 }


experiment4["best_rmse"]=train_and_val_model(model4, optimizer4, torch.nn.MSELoss() , experiment4["data_aug"], 
                                            train_set, experiment4["train_batch_size"], val_set, experiment4["val_batch_size"] ,
                                            experiment4["num_epochs"], experiment4["params_path"],experiment4["train_rmse"], experiment4["val_rmse"])

#storing results of the experiment
experiment_store(experiment4,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 5 (data aug + 3 unfrozen)
model5=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model5.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model5=model_frozen(model5,n_unfreeze=3).to('cuda')

optimizer5=torch.optim.SGD(model5.parameters(), lr=0.001, momentum=0.9)


experiment5={"exp_number": 5, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.4),
                                  transforms.RandomHorizontalFlip(p=0.4),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.4,0.2,0.2,0.2]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp5_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 3 }


experiment5["best_rmse"]=train_and_val_model(model5, optimizer5, torch.nn.MSELoss() , experiment5["data_aug"], 
                                            train_set, experiment5["train_batch_size"], val_set, experiment5["val_batch_size"] ,
                                            experiment5["num_epochs"], experiment5["params_path"], experiment5["train_rmse"], experiment5["val_rmse"])

#storing results of the experiment
experiment_store(experiment5,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 6 (data aug + 4 unfrozen)
model6=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model6.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model6=model_frozen(model6,n_unfreeze=4).to('cuda')

optimizer6=torch.optim.SGD(model6.parameters(), lr=0.001, momentum=0.9)


experiment6={"exp_number": 6, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.4),
                                  transforms.RandomHorizontalFlip(p=0.4),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.4,0.2,0.2,0.2]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp6_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 4 }


experiment6["best_rmse"]=train_and_val_model(model6, optimizer6, torch.nn.MSELoss() , experiment6["data_aug"], 
                                            train_set, experiment6["train_batch_size"], val_set, experiment6["val_batch_size"] ,
                                            experiment6["num_epochs"], experiment6["params_path"], experiment6["train_rmse"], experiment6["val_rmse"])

#storing results of the experiment
experiment_store(experiment6,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 7 (no frozen layers + data aug without rotating images)
model7=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model7.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model7.to('cuda')

optimizer7=torch.optim.SGD(model7.parameters(), lr=0.001, momentum=0.9)


experiment7={"exp_number": 7, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.4),
                                  transforms.RandomHorizontalFlip(p=0.4),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp7_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment7["best_rmse"]=train_and_val_model(model7, optimizer7, torch.nn.MSELoss() , experiment7["data_aug"], 
                                            train_set, experiment7["train_batch_size"], val_set, experiment7["val_batch_size"] ,
                                            experiment7["num_epochs"], experiment7["params_path"],experiment7["train_rmse"], experiment7["val_rmse"])

#storing results of the experiment
experiment_store(experiment7,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 8 (0 frozen + data aug with different parameters)
model8=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model8.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model8.to('cuda')

optimizer8=torch.optim.SGD(model8.parameters(), lr=0.001, momentum=0.9)


experiment8={"exp_number": 8, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.25,0.25,0.25,0.25]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp8_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment8["best_rmse"]=train_and_val_model(model8, optimizer8, torch.nn.MSELoss() , experiment8["data_aug"], 
                                            train_set, experiment8["train_batch_size"], val_set, experiment8["val_batch_size"] ,
                                            experiment8["num_epochs"], experiment8["params_path"],experiment8["train_rmse"], experiment8["val_rmse"])

#storing results of the experiment
experiment_store(experiment8,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 9 (0 frozen + data aug with different parameters + smaller batch size)
model9=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model9.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model9.to('cuda')

optimizer9=torch.optim.SGD(model9.parameters(), lr=0.001, momentum=0.9)


experiment9={"exp_number": 9, "train_batch_size": 250 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.25,0.25,0.25,0.25]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp9_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment9["best_rmse"]=train_and_val_model(model9, optimizer9, torch.nn.MSELoss() , experiment9["data_aug"], 
                                            train_set, experiment9["train_batch_size"], val_set, experiment9["val_batch_size"] ,
                                            experiment9["num_epochs"], experiment9["params_path"],experiment9["train_rmse"], experiment9["val_rmse"])

#storing results of the experiment
experiment_store(experiment9,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 10 (no frozen + data aug + Adam optimizer)
model10=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model10.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model10.to('cuda')

optimizer10=torch.optim.Adam(model10.parameters(), lr=0.001)


experiment10={"exp_number": 10, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.25,0.25,0.25,0.25]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp10_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment10["best_rmse"]=train_and_val_model(model10, optimizer10, torch.nn.MSELoss() , experiment10["data_aug"], 
                                            train_set, experiment10["train_batch_size"], val_set, experiment10["val_batch_size"] ,
                                            experiment10["num_epochs"], experiment10["params_path"],experiment10["train_rmse"], experiment10["val_rmse"])

#storing results of the experiment
experiment_store(experiment10,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 11 (no frozen + data aug + lr increased to 0.01)
model11=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model11.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model11.to('cuda')

optimizer11=torch.optim.SGD(model11.parameters(), lr=0.01, momentum=0.9)


experiment11={"exp_number": 11, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.25,0.25,0.25,0.25]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp11_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment11["best_rmse"]=train_and_val_model(model11, optimizer11, torch.nn.MSELoss() , experiment11["data_aug"], 
                                            train_set, experiment11["train_batch_size"], val_set, experiment11["val_batch_size"] ,
                                            experiment11["num_epochs"], experiment11["params_path"],experiment11["train_rmse"], experiment11["val_rmse"])

#storing results of the experiment
experiment_store(experiment11,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 12 (no frozen + data aug + Adam optimizer + better learning rate)
model12=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model12.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model12.to('cuda')

optimizer12=torch.optim.Adam(model12.parameters(), lr=0.01)


experiment12={"exp_number": 12, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.25,0.25,0.25,0.25]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp12_params.pt" ,"best_rmse": 0, "t rain_rmse": [], "val_rmse": [], "frozen": 0 }


experiment12["best_rmse"]=train_and_val_model(model12, optimizer12, torch.nn.MSELoss() , experiment12["data_aug"], 
                                            train_set, experiment12["train_batch_size"], val_set, experiment12["val_batch_size"] ,
                                            experiment12["num_epochs"], experiment12["params_path"],experiment12["train_rmse"], experiment12["val_rmse"])

#storing results of the experiment
experiment_store(experiment12,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 13 (no frozen + data aug without gaussian blur)
model13=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model13.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model13.to('cuda')

optimizer13=torch.optim.SGD(model13.parameters(), lr=0.001, momentum=0.9)


experiment13={"exp_number": 13, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.25,0.25,0.25,0.25])]), 
              "num_epochs": 100, 
              "params_path": "data/pytorch/exp13_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment13["best_rmse"]=train_and_val_model(model13, optimizer13, torch.nn.MSELoss() , experiment13["data_aug"], 
                                            train_set, experiment13["train_batch_size"], val_set, experiment13["val_batch_size"] ,
                                            experiment13["num_epochs"], experiment13["params_path"],experiment13["train_rmse"], experiment13["val_rmse"])

#storing results of the experiment
experiment_store(experiment13,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 14 (no frozen + data aug + lr increased to 0.01 + smaller batch size)
model14=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model14.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model14.to('cuda')

optimizer14=torch.optim.SGD(model14.parameters(), lr=0.01, momentum=0.9)


experiment14={"exp_number": 14, "train_batch_size": 250 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.25,0.25,0.25,0.25]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp14_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment14["best_rmse"]=train_and_val_model(model14, optimizer14, torch.nn.MSELoss() , experiment14["data_aug"], 
                                            train_set, experiment14["train_batch_size"], val_set, experiment14["val_batch_size"] ,
                                            experiment14["num_epochs"], experiment14["params_path"],experiment14["train_rmse"], experiment14["val_rmse"])

#storing results of the experiment
experiment_store(experiment14,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 15 (no frozen + data aug + lr increased to 0.01 + no momentum)
model15=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model15.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model15.to('cuda')

optimizer15=torch.optim.SGD(model15.parameters(), lr=0.01)


experiment15={"exp_number": 15, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.25,0.25,0.25,0.25]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp15_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment15["best_rmse"]=train_and_val_model(model15, optimizer15, torch.nn.MSELoss() , experiment15["data_aug"], 
                                            train_set, experiment15["train_batch_size"], val_set, experiment15["val_batch_size"] ,
                                            experiment15["num_epochs"], experiment15["params_path"],experiment15["train_rmse"], experiment15["val_rmse"])

#storing results of the experiment
experiment_store(experiment15,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 16 (no frozen + data aug + lr increased to 0.01 + small momentum )
model16=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model16.fc=torch.nn.Linear(512, 1)
#unfreezing the last conv2d layer
model16.to('cuda')

optimizer16=torch.optim.SGD(model16.parameters(), lr=0.01, momentum=0.5)


experiment16={"exp_number": 16, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.25,0.25,0.25,0.25]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp16_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment16["best_rmse"]=train_and_val_model(model16, optimizer16, torch.nn.MSELoss() , experiment16["data_aug"], 
                                            train_set, experiment16["train_batch_size"], val_set, experiment16["val_batch_size"] ,
                                            experiment16["num_epochs"], experiment16["params_path"],experiment16["train_rmse"], experiment16["val_rmse"])

#storing results of the experiment
experiment_store(experiment16,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 17 (training a bit more experiment 11)
model17=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model17.fc=torch.nn.Linear(512, 1)
#loading parameters from model 11
model17.load_state_dict(torch.load("data/pytorch/exp11_params.pt"))

#unfreezing the last conv2d layer
model17.to('cuda')

optimizer17=torch.optim.SGD(model17.parameters(), lr=0.01, momentum=0.9)


experiment17={"exp_number": 17, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.25,0.25,0.25,0.25]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp17_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment17["best_rmse"]=train_and_val_model(model17, optimizer17, torch.nn.MSELoss() , experiment17["data_aug"], 
                                            train_set, experiment17["train_batch_size"], val_set, experiment17["val_batch_size"] ,
                                            experiment17["num_epochs"], experiment17["params_path"],experiment17["train_rmse"], experiment17["val_rmse"])

#storing results of the experiment
experiment_store(experiment17,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#experiment 18: re doing experiment 10
model18=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model18.fc=torch.nn.Linear(512, 1)
#loading parameters from model 10
model18.load_state_dict(torch.load("data/pytorch/exp10_params.pt"))
#unfreezing the last conv2d layer
model18.to('cuda')

optimizer18=torch.optim.Adam(model18.parameters(), lr=0.001)


experiment18={"exp_number": 18, "train_batch_size": 500 , "val_batch_size": 1243,
              "data_aug": transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  rotation_discrete(angles=[0,90,180,270],probs=[0.25,0.25,0.25,0.25]),
                                  transforms.GaussianBlur(kernel_size=(9,9))]), "num_epochs": 100, 
              "params_path": "data/pytorch/exp18_params.pt" ,"best_rmse": 0, "train_rmse": [], "val_rmse": [], "frozen": 0 }


experiment18["best_rmse"]=train_and_val_model(model18, optimizer18, torch.nn.MSELoss() , experiment18["data_aug"], 
                                            train_set, experiment18["train_batch_size"], val_set, experiment18["val_batch_size"] ,
                                            experiment18["num_epochs"], experiment18["params_path"],experiment18["train_rmse"], experiment18["val_rmse"])

#storing results of the experiment
experiment_store(experiment18,"data/results/results.csv","data/results/results_train.csv","data/results/results_val.csv")

#for our best model, we compute final metrics and predict for test set.
final_metrics_predict(model18,torch.nn.MSELoss(),torchmetrics.R2Score().to('cuda'),train_set,experiment18["train_batch_size"],
                       val_set,test_set,experiment18["params_path"],"data/processed/test_set_only.csv","data/processed/test_set_with_pred.csv",
                       "data/results/results_best_exp.csv")