To run the code, one should open the terminal in the process directory and :

For Training : 
     python3 train.py model{model_number} {classifier} 
    
    in which model_number is the model number (goes from 1...4) and classifier is the type of classifier (one should only choose "svm" for now).

For Testing : 
    python3 test.py {model_filename} {results_filename} {prob1, prob2, etc...}

    in which model_filename is the filename where the serialized model from the training phase is stored, results_filename is the name of the file where the results of the test are to be stored, and prob1, prob2 is the probabilities that we want to be tested out.