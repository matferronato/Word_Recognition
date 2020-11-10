

from machineLearnTrain import trainML
from machineLearnTest import testML
from audioRecord import userRecord


def main():
    
    print("run machine learning? y/n")
    userInput = input() 
    if "y" in userInput.casefold():
        trainML()
    else:
        while(True):
            print("input something when you are ready to speak")
            input()         
            userRecord()
            testML()
        
if __name__ == '__main__': # chamada da funcao principal
    main()