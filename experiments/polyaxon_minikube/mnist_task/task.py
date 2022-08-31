from polyaxon import tracking
import argparse
import logging
import numpy as np
import time
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(__file__ ,"../../../../../"))
sys.path.append(PROJECT_ROOT)
from ml_benchmark.workload.mnist.mnist_task import MnistTask


def train(times ,epoch):
    loss = 1
    for x in np.arange(1,times + 1): 
        loss = 1 - (x -1 ) / times
        time.sleep(0.1)
        msg = "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                epoch, x , times,
                100. * x / times, loss)
        logging.info(msg)


def test():

    test_accuracy =0.8 
    logging.info("Validation-accuracy={:.4f}\n".format(
        test_accuracy))



def main():
   
    
    #parsing arguments 
    parser = argparse.ArgumentParser(description="MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                        help="learning rate (default: 0.01)")
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr






    #MnistTask
    task = MnistTask(config_init={"epochs": epochs})
    objective = task.create_objective()
    #TODO add the weight decay to the definition of the template
    objective.set_hyperparameters({"learning_rate": lr, "weight_decay": 0.01})
    objective.train()
    validation_scores = objective.validate()
    
  
    #Geting results
    avg = validation_scores["weighted avg"]
    print("precision",avg["precision"])
    print("f1-score",avg["f1-score"])

    # logging.basicConfig(
    #         format="%(asctime)s %(levelname)-8s %(message)s",
    #         datefmt="%Y-%m-%dT%H:%M:%SZ",
    #         level=logging.DEBUG)
    # for epoch in np.arange(1,epochs+1):
    #     train(batch_size,epoch)
    #     test() 

    # Polyaxon
    #initiating polyaxon tracking
    tracking.init()    
    #loging metrics 
    tracking.log_metrics(recall=avg["recall"], )
    #logging the results 
    tracking.log_outputs(f1score=avg["f1-score"],precision=avg["precision"])


if __name__ == "__main__":
    main()
