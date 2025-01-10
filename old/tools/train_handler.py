import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
import os
import matplotlib.pyplot as plt
import json

## loss functie meet hoe de voorspeling overeenkomt met ground truth
## optimizer voert backward pass uit
## "Providing feedback (loss) and a strategy for adjusting weights (optimization)."

def __make_directory() -> str:
    print("Making directory...")
    try:
        now = datetime.now().strftime("%Y_%m_%d %H_%M")
        new_folder_path = f"models/model at {now}/"
        os.mkdir(new_folder_path)
        return new_folder_path
    except Exception as e:
        print(f"Failed to make directory: {e}")
        return ""
    

def trainer(model, train_loader:DataLoader, loss_func, optimizer, epochs) -> str:
    print("Training...")
    new_folder_path = __make_directory()
    ## Zet model in train stand hoeft niets te returnen omdat de class mutable is en arguments references zijn
    model.train()
    ## Ga voor elke epoch...
    for epoch in range(epochs):
        epoch_loss = 0.0
        print("Epoch: ", epoch + 1)
        ## ...langs alle data (welke ieder een batch is van x aantal regels)
        for batch, x in train_loader:
            print(batch, x)
            resume_embedding, desc_embedding, labels = [tensor.to(model.device) for tensor in batch]
            
            predictions = model(
                resume_embedding = resume_embedding, 
                desc_embedding = desc_embedding
            )

            ## targets/labels en outputs flattenen en als float opslaan.
            ## Werkt dit ook weer zonder .float()?
            ran_labels = torch.squeeze(labels.float())
            ran_predictions = torch.squeeze(predictions.float())

            ## Fout in voorspellingen berekenen
            loss = loss_func(input=ran_predictions, target=ran_labels)
            epoch_loss += loss.item()
            print(f"batch loss: {round(loss.item(),3)}")
            
            ## Cleared vorige leerresultaten
            optimizer.zero_grad()
            ## Gewichten en bias aanpassen in NN
            ## "Calculate how changing the weight slightly would affect the loss."
            ## TODO uitzoeken chainrule
            ## TODO vragen tony backpropagation
            ## TODO uitleg backprop in schrift zetten
            ## "The framework of the model builds a 'computation graph' which allows the loss to know the weights."
            loss.backward()
            ## Update de weights
            optimizer.step()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"epoch loss : {round(avg_epoch_loss,3)}")
    
    torch.save(model.state_dict(), new_folder_path + "trained_model.pth")
    print("Training done")
    return new_folder_path


def evaluation(model:nn.Module, test_loader:DataLoader, loss_func, new_folder_path:str, config:dict) -> None:
    print("Evaluating...")
    all_predictions = []
    all_labels = []
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for resume_ids, resume_mask, desc_ids, desc_mask, labels in test_loader:

            predictions = model(
                resume_ids = resume_ids, 
                resume_mask = resume_mask, 
                desc_ids = desc_ids, 
                desc_mask = desc_mask
            )
            
            ## Squeezen tot (altijd) 1 dimensie 
            ran_labels = torch.squeeze(labels.float())
            ran_predictions = torch.squeeze(predictions.float())

            ## Loss berekenen en optellen
            loss = loss_func(ran_predictions, ran_labels)
            total_loss += loss.item()

            ## Opslaan labels en predictions
            labels_type_checked = ran_labels.tolist() if ran_labels.dim() != 0 else [ran_labels.item()]
            all_labels += labels_type_checked
            predictions_type_checked = ran_predictions.tolist() if ran_predictions.dim() != 0 else [ran_predictions.item()]
            predictions_rounded = [round(x, 3) for x in predictions_type_checked]
            all_predictions += predictions_rounded

            # all_predictions.extend(output_score.cpu().numpy().flatten()) ## FIXME: difference?
            # all_targets.extend(target_tensor.cpu().numpy().flatten())
            # all_predictions.extend(torch.argmax(output_score, dim=1).cpu().numpy())
            # all_targets.extend(target_score.cpu().numpy())

    ## Save evaluation
    with open(new_folder_path + "data.json", "w" )as file:
        json.dump({
            "config" : config,

            "evaluation_data" : {
                "all_labels" : all_labels,
                "all_predictions" : all_predictions
            }
        },
        fp = file
        )

    __visualize_evaluation(all_labels=all_labels, all_predictions=all_predictions, new_folder_path=new_folder_path)


    print(f"Total loss: {round(total_loss,3)}")
    return all_labels, all_predictions

def __visualize_evaluation(all_labels:list, all_predictions:list, new_folder_path:str):

    plt.figure(figsize=(10,7))
    plt.scatter(x=all_labels, y=all_predictions)
    plt.savefig(new_folder_path + "scatter.png")