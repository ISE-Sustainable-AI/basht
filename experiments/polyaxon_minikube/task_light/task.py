from __future__ import print_function

from polyaxon import tracking
import argparse
import logging
import os
import numpy as np
import time
from ml_benchmark.latency_tracker import latency_decorator

@latency_decorator
def train(times ,epochs):
    loss = 1
    for epoch in range(epochs):
        for x in np.arange(1,times + 1): 
            loss = 1 - (x -1 ) / times
            time.sleep(0.01)
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
  
    #timestamps format
    logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG)

   
    #model taining and testing
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    
    train(batch_size,epochs)
    test()
    avg = {"recall":0.5,"f1-score":0.98,"precision":0.7} 
    # Polyaxon
    #initiating polyaxon tracking
    tracking.init()    
    #loging metrics 
    tracking.log_metrics(recall=avg["recall"], )
    #logging the results 
    tracking.log_outputs(f1score=avg["f1-score"],precision=avg["precision"]) 



if __name__ == "__main__":
    main()
