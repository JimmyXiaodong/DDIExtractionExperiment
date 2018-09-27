'''
This script read the train and test data from .txt file
and save it four different lists.
'''

# Read the train text data, and then save the data in train_text
file_DrugBank_text = open("Jimmy_DDIE_Train_Data_Processed/DDIE2013_Train_DrugBank_text.txt")
train_text = file_DrugBank_text.readlines()

file_Medline_text = open("Jimmy_DDIE_Train_Data_Processed/DDIE2013_Train_Medline_text.txt")
train_text.extend(file_Medline_text.readlines())

file_DrugBank_text.close()
file_Medline_text.close()

# Read the train label, and then save the label in train_label
# Map the label to the corresponding ID
file_DrugBank_label = open("Jimmy_DDIE_Train_Data_Processed/DDIE2013_Train_DrugBank_label.txt")
train_label = file_DrugBank_label.readlines()

file_Medline_label = open("Jimmy_DDIE_Train_Data_Processed/DDIE2013_Train_Medline_label.txt")
train_label.extend(file_Medline_label.readlines())

file_DrugBank_label.close()
file_Medline_label.close()

# Read the test text data, and then save the data in test_text
file_DrugBank_text = open("Jimmy_DDIE_Test_Data_Processed/DDIE2013_Test_DrugBank_text.txt")
test_text = file_DrugBank_text.readlines()

file_Medline_text = open("Jimmy_DDIE_Test_Data_Processed/DDIE2013_Test_Medline_text.txt")
test_text.extend(file_Medline_text.readlines())

file_DrugBank_text.close()
file_Medline_text.close()

# Read the test label, and then save the label in train_label
# Map the label to the corresponding ID
file_DrugBank_label = open("Jimmy_DDIE_Test_Data_Processed/DDIE2013_Test_DrugBank_label.txt")
test_label = file_DrugBank_label.readlines()

file_Medline_label = open("Jimmy_DDIE_Test_Data_Processed/DDIE2013_Test_Medline_label.txt")
test_label.extend(file_Medline_label.readlines())

file_DrugBank_label.close()
file_Medline_label.close()

train_label_ID = []
test_label_ID = []


for label in train_label:
    label = label.strip('\n')
    if label == 'false' or label == 'null':
        train_label_ID.append(0)
    elif label == 'int':
        train_label_ID.append(1)
    elif label == 'advise':
        train_label_ID.append(2)
    elif label == 'effect':
        train_label_ID.append(3)
    else:
        train_label_ID.append(4)

for label in test_label:
    label = label.strip('\n')
    if label == 'false'or label == 'null':
        test_label_ID.append(0)
    elif label == 'int':
        test_label_ID.append(1)
    elif label == 'advise':
        test_label_ID.append(2)
    elif label == 'effect':
        test_label_ID.append(3)
    else:
        test_label_ID.append(4)

for ids in test_label_ID:
    if ids!=1 and ids!=2 and ids!=3 and ids!=4 and ids!=0:
        print ("hahha")
