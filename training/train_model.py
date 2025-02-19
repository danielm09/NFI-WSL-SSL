import wandb
import time
import torch



def trainWSLModel(model, train_loader, test_loader, optimizer, criterion, test_eval=True, mask_pixel=20, log_to_wandb=True, wandb_proj=None, run_config=None, save=True):

    """
    Start model training.

    Parameters:
    model: pytorch model;
    train_loader: DataLoader object with training data;
    test_loader: DataLoader object with test data (required if test_eval=True, otherwise parse None);
    optimizer: pytorch optimizer;
    criterion: loss criterion;
    test_eval: boolean. If True, computes accuracy metrics for the test set after each training epoch;
    mask_pixel: pixels equal to mask_pixel will be masked for accuracy evaluation purposes (use it to mask unlabeled pixels);
    log_to_wandb: boolean. determines whether run details should be logged to wandb;
    wandb_proj: wandb project name;
    run_config: dictionary with configurations of the current run (must have a least an 'epochs' key);
    save: boolean. determines whether to save the model at the end of all epochs.

    Returns: None
    """

    if log_to_wandb:
        wandb.init(project=wandb_proj, config=run_config)

    #send model to device
    model = model.cuda()
    
    print("Training model")

    for epoch in range(run_config['epochs']):

        tstart = time.time()

        model.train()
        #training metrics
        running_loss = 0.0
        correct = 0
        total = 0

        #loop through training batches
        for inputs, labels in train_loader:
            # Transfer to GPU 
            inputs, labels = inputs.cuda(), labels.cuda()
            # Zero the parameter gradients
            optimizer.zero_grad()
            #forward pass
            outputs = model(inputs)
            #compute loss
            loss = criterion(outputs, labels)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, dim=1)

            if mask_pixel: #calculates correct predictions ignoring pixels equal to mask_pixel
                mask = labels!=mask_pixel
                labels_selected = labels[mask]
                predicted_selected = predicted[mask]
                correct += (predicted_selected == labels_selected).sum().item()  # Sum the correct predictions
                total += labels_selected.size(0)
            else: #calculates correct prediction for all pixels
                correct += (predicted==labels).sum().item()
                total += labels.size(0)
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)

        #evaluate accuracy on test
        if test_eval: 
            val_loss, val_accuracy = evaluate(model, test_loader, criterion)
        else: #report test loss and accuracy as zero
            val_loss, val_accuracy = (0, 0)

        #log metrics to wandb
        if log_to_wandb:
            wandb.log({"epoch": epoch, "train_loss": avg_loss, "train_acc":correct/total, "val_loss":val_loss, "val_acc":val_accuracy})

        print(f"Epoch [{epoch+1}/{run_config['epochs']}], Average Loss: {avg_loss:.4f}, Acc: {correct / total :.4f}, Test_eval: {str(test_eval)}, Test Acc: {val_accuracy:.4f}, Time/epoch: {round((time.time()-tstart)/60,2)}min")
        
    #save model
    if save:
        #output folder is models/saved_models
        out_folder = "../models/saved_models/" #this is a relative path, considering the function will be called from the notebook
        #the model name requires the following dictionary structure
        model_name = f"ModelWSL_{run_config['architecture']}_depths{run_config['depths'].replace(', ','-')}_dims{run_config['dims'].replace(', ','-')}_batch{run_config['batch_size']}_lr{str(run_config['learning_rate'])[2:]}_Aug{run_config['augmentations']}_{run_config['optimizer']}_{run_config['criterion']}.pt"
        torch.save(model, out_folder + model_name) #saves model after last epoch

    #finishes wandb run
    if log_to_wandb:
        wandb.finish()  

def evaluate(model, dataloader, criterion, mask_pixel=20):
    """
    Evaluates model accuracy on test dataset.

    Parameters:
    model: pytorch model;
    dataloader: DataLoader object with the test set;
    criterion: loss criterion;
    mask_pixel: pixels equal to mask_pixel will be masked for accuracy evaluation purposes (use it to mask unlabeled pixels).

    Returns: tuple(average test set loss, test set accuracy)
    """

    model.eval()  # Set the model to evaluation mode
    #test metrics
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in dataloader:

            inputs, labels = inputs.cuda(), labels.cuda()
            
            #predict labels
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)

            if mask_pixel: #calculates correct predictions ignoring pixels equal to mask_pixel
                mask = labels!=20
                labels_selected = labels[mask]
                predicted_selected = predicted[mask]
                correct += (predicted_selected == labels_selected).sum().item()  # Sum the correct predictions
                total += labels_selected.size(0)
            else: #calculates correct predictions to all pixels
                correct += (predicted==labels).sum().item()
                total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy

 
def trainMAE(model, train_loader, optimizer, mask_ratio, scheduler=None, log_to_wandb=True, wandb_proj=None, run_config=None, save=True):
    """
    Runs the Self-Supervised Learning MAE pretraining.

    Parameters:
    model: pytorch model;
    train_loader: DataLoader object with training data;
    optimizer: pytorch optimizer;
    mask_ratio: ratio of masked patches;
    scheduler: learning rate scheduler;
    log_to_wandb: boolean. defines whether to log run details to wandb;
    wandb_proj: wand project name;
    run_config: dictionary with run configs;
    save: boolean. defines whether to save the model after training.
    """

    if log_to_wandb:
        wandb.init(project=wandb_proj, config=run_config)

    #send model to device
    model = model.cuda()
    
    print("Training model")

    for epoch in range(run_config['epochs']):

        tstart = time.time()
        model.train()
        running_loss = 0.0

        for inputs in train_loader:
            # Transfer to GPU 
            inputs = inputs.cuda()#, labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute predictions and loss
            loss, _, _ = model(inputs, None, mask_ratio=mask_ratio)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        if scheduler:
            scheduler.step()

        if log_to_wandb:
            wandb.log({"epoch": epoch, "train_loss": avg_loss, "lr": optimizer.param_groups[0]['lr']})

        print(f"Epoch [{epoch+1}/{run_config['epochs']}], Average Loss: {avg_loss:.6f}, Time/epoch: {round((time.time()-tstart)/60,2)}min")

    #save model
    if save:
        #output folder is models/saved_models
        out_folder = "../models/saved_models/" #this is a relative path, considering the function will be called from the notebook
        #the model name requires the following dictionary structure
        model_name = f"MAEModel_{run_config['architecture']}_depths{run_config['depths'].replace(', ','-')}_dims{run_config['dims'].replace(', ','-')}_batch{run_config['batch_size']}_lr{str(run_config['learning_rate'])[2:]}_Aug{run_config['augmentations']}_{run_config['optimizer']}_{run_config['criterion']}.pt"
        torch.save(model.state_dict(), out_folder + model_name) #saves model (parameters only) after last epoch

    #finishes wandb run
    if log_to_wandb:
        wandb.finish()
