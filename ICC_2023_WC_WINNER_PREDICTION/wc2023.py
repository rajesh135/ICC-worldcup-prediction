import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


import tkinter
from tkinter import *
from tkinter import messagebox
root=Tk()
root.geometry('1024x720')
root.configure(bg='white')


l1 = Label(root,text="import result.csv file",font='Helvetica 14 bold',fg='tomato').grid(row=0,column=0)

def result():
	global results
	results1=filedialog.askopenfilename(initialdir="/home/bahubali/MCA-3/ML/Assignment/ICC-2023-WC/datasets_modified",title="select a file",filetypes =(("csv File", "*.csv"),("All Files","*.*")))
	results=pd.read_csv(results1)
resultbutton=Button(root,text="import",width=15,activebackground="green",command=result).grid(row=0,column=2,padx=40,pady=0)

############################################################################################################################################


l2 = Label(root,text="import World Cup 2023 Dataset.csv",font='Helvetica 14 bold',fg='tomato').grid(row=4,column=0)

def world_cup():
	global world_cup
	world_cup1=filedialog.askopenfilename(initialdir="/home/bahubali/MCA-3/ML/Assignment/ICC-2023-WC/datasets_modified",title="select a file",filetypes =(("csv File", "*.csv"),("All Files","*.*")))
	world_cup=pd.read_csv(world_cup1)

world_cup_tbutton=Button(root,text="import",width=15,activebackground="green",command=world_cup).grid(row=4,column=2,padx=40,pady=0)

############################################################################################################################################



l3 = Label(root,text="import icc_rankings.csv",font='Helvetica 14 bold',fg='tomato').grid(row=8,column=0)

def icc_ranking():
	global ranking
	ranking1=filedialog.askopenfilename(initialdir="/home/bahubali/MCA-3/ML/Assignment/ICC-2023-WC/datasets_modified",title="select a file",filetypes =(("csv File", "*.csv"),("All Files","*.*")))
	ranking=pd.read_csv(ranking1)

icc_ranking_tbutton=Button(root,text="import",width=15,activebackground="green",command=icc_ranking).grid(row=8,column=2,padx=40,pady=0)


############################################################################################################################################



l4 = Label(root,text="import fixtures.csv",font='Helvetica 14 bold',fg='tomato').grid(row=12,column=0)

def fixture():
	global fixtures1
	fixtures2=filedialog.askopenfilename(initialdir="/home/bahubali/MCA-3/ML/Assignment/ICC-2023-WC/datasets_modified",title="select a file",filetypes =(("csv File", "*.csv"),("All Files","*.*")))
	fixtures1=pd.read_csv(fixtures2)

fixtures_tbutton=Button(root,text="import",width=15,activebackground="green",command=fixture).grid(row=12,column=2,padx=40,pady=0)


############################################################################################################################################


def train():
#	fixtures=filedialog.askopenfilename(initialdir="home",title="select a file",filetypes =(("csv File", "*.csv"),("All Files","*.*")))
#	print(fixtures)
	global df,india,india_2010,worldcup_teams,df_teams_1,df_teams_2,df_teams,df1_teams_2010

	df = results[(results['Team_1'] == 'India') | (results['Team_2'] == 'India')]
	india = df.iloc[:]

	year = []
	for row in india['date']:
		year.append(int(row[7:]))
	india ['match_year']= year
#print(year)
#print(india)
	india_2010 = india[india.match_year >= 10]



	worldcup_teams = ['England', ' South Africa', '', 'West Indies', 
            'Pakistan', 'New Zealand', 'Sri Lanka', 'Afghanistan', 
            'Australia', 'Bangladesh', 'India']
	df_teams_1 = results[results['Team_1'].isin(worldcup_teams)]
	df_teams_2 = results[results['Team_2'].isin(worldcup_teams)]
	df_teams = pd.concat((df_teams_1, df_teams_2))
	df_teams.drop_duplicates()



	l5a = Label(root,text="model is being trained",font='Helvetica 14 bold').grid(row=20,column=0)


l5 = Label(root,text="clean and train the model",font='Helvetica 14 bold',fg='tomato').grid(row=16,column=0)

train_tbutton=Button(root,text="Clean data",width=15,activebackground="green",command=train).grid(row=16,column=2,padx=40,pady=0)


############################################################################################################################################
def build_model():

	global final,X,y,X_train, X_test, y_train, y_test,df_teams_2010


	df1_teams_2010 = df_teams.drop(['date'], axis=1)
	#df1_teams_2010.head()
	#print(df_teams_2010)
	print(df1_teams_2010.count())

	#dropping rows if contains NA as a result
	df_teams_2010=df1_teams_2010.dropna(axis=0,how='any')
	#print(df_teams_2010.count())




	df_teams_2010 = df_teams_2010.reset_index(drop=True)
	df_teams_2010.loc[df_teams_2010.Winner == df_teams_2010.Team_1,'winning_team']=1
	df_teams_2010.loc[df_teams_2010.Winner == df_teams_2010.Team_2, 'winning_team']=2

	df_teams_2010.loc[df_teams_2010.Margin == df_teams_2010.Team_1, 'winning_team']=3
	df_teams_2010.loc[df_teams_2010.Margin == df_teams_2010.Team_2, 'winning_team']=4

	df_teams_2010.loc[df_teams_2010.Ground == df_teams_2010.Team_1, 'winning_team']=5
	df_teams_2010.loc[df_teams_2010.Margin == df_teams_2010.Team_2, 'winning_team']=6
	df_teams_2010 = df_teams_2010.drop(['winning_team'], axis=1)

	#	df_teams_2010.head()

	#convert team-1 and team-2 from categorical variables to continous inputs 
	# Get dummy variables
	final = pd.get_dummies(df_teams_2010, prefix=['Team_1', 'Team_2','Margin', 'Ground'], columns=['Team_1', 'Team_2','Margin', 'Ground'])

	# Separate X and y sets
	X = final.drop(['Winner'], axis=1)
	y = final["Winner"]


	# Separate train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


	l6a = Label(root,text="Model built is done",font='Helvetica 14 bold').grid(row=28,column=0)


l6 = Label(root,text="Building a model",font='Helvetica 14 bold',fg='tomato').grid(row=24,column=0)

build_model_tbutton=Button(root,text="Build a Model",width=15,activebackground="green",command=build_model).grid(row=24,column=2,padx=40,pady=0)

############################################################################################################################################

def Random_Forest():
	global rf,pred_set,backup_pred_set,missing_cols,fixtures
	global score 
	#score= DoubleVar()
	global score2
	#score2= DoubleVar()

	rf = RandomForestClassifier(n_estimators=100, max_depth=20,random_state=0)
	rf.fit(X_train, y_train) 


	score = rf.score(X_train, y_train)
	score2 = rf.score(X_test, y_test)


	print("Training set accuracy: ", '%.3f'%(score))
	print("Test set accuracy: ", '%.3f'%(score2))
	print("\n\n")
	#StringVar =str(scorre)
	a=str('%.3f'%(score))
	b=str('%.3f'%(score2))

	fixtures = fixtures1.drop(['Round Number','Date','Group '], axis=1)
	pred_set = []

	# Create new columns with ranking position of each team
	fixtures.insert(1, 'first_position', fixtures['Team_1'].map(ranking.set_index('Team')['Position']))
	fixtures.insert(2, 'second_position', fixtures['Team_2'].map(ranking.set_index('Team')['Position']))

	# We only need the group stage games, so we have to slice the dataset
	fixtures = fixtures.iloc[:45, :]


	# Loop to add teams to new prediction dataset based on the ranking position of each team
	for index, row in fixtures.iterrows():
	    if row['first_position'] < row['second_position']:
	        pred_set.append({'Team_1': row['Team_1'], 'Team_2': row['Team_2'], 'winning_team': None})
	    else:
	        pred_set.append({'Team_1': row['Team_2'], 'Team_2': row['Team_1'], 'winning_team': None})
	        
	pred_set = pd.DataFrame(pred_set)
	backup_pred_set = pred_set


	# Get dummy variables and drop winning_team column
	pred_set = pd.get_dummies(pred_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])

	# Add missing columns compared to the model's training dataset
	missing_cols = set(final.columns) - set(pred_set.columns)
	for c in missing_cols:
	    pred_set[c] = 0
	pred_set = pred_set[final.columns]
	#print(pred_set)

	pred_set = pred_set.drop(['Winner'], axis=1)




	l7a = Label(root,text="Training set accuracy is ",font='Helvetica 14 bold').grid(row=36,column=0)
	l7a_a = Label(root,text=a,font='Helvetica 14 bold',width=15).grid(row=36,column=2)
	l7b = Label(root,text="Test set accuracy is ",font='Helvetica 14 bold').grid(row=40,column=0)
	l7b_b = Label(root,text=b,font='Helvetica 14 bold',width=15).grid(row=40,column=2)



l7 = Label(root,text="Apply Random Forest Algorithm",font='Helvetica 14 bold',fg='tomato').grid(row=32,column=0)

build_model_button=Button(root,text="Apply",width=15,activebackground="green",command=Random_Forest).grid(row=32,column=2,padx=40,pady=0)


############################################################################################################################################

def league_matches():
	global predictions,semi
	#group matches 
	predictions = rf.predict(pred_set)
	for i in range(fixtures.shape[0]):
	    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
	    if predictions[i] == 1:
	        print("Winner: " + backup_pred_set.iloc[i, 1])

	    else:
	        print("Winner: " + backup_pred_set.iloc[i, 0])
	    print("")


	# List of tuples before 
	print("\n\n")
	semi = [('New Zealand', 'India'),('England', 'South Africa')]



l8 = Label(root,text="Find league matches result",font='Helvetica 14 bold',fg='tomato').grid(row=44,column=0)

league_match_button=Button(root,text="League matches",width=15,activebackground="green",command=league_matches).grid(row=44,column=2,padx=40,pady=0)


############################################################################################################################################

def clean_and_predict1(matches, ranking, final, logreg):
	global positions

#    # Initialization of auxiliary list for data cleaning
	positions = []

#    # Loop to retrieve each team's position according to ICC ranking
	for match in matches:
		positions.append(ranking.loc[ranking['Team'] == match[0],'Position'].iloc[0])
		positions.append(ranking.loc[ranking['Team'] == match[1],'Position'].iloc[0])
    
#   # Creating the DataFrame for prediction
	pred_set = []

#    # Initializing iterators for while loop
	i = 0
	j = 0

#    # 'i' will be the iterator for the 'positions' list, and 'j' for the list of matches (list of tuples)
	while i < len(positions):
		dict1 = {}

#       # If position of first team is better then this team will be the 'Team_1' team, and vice-versa
		if positions[i] < positions[i + 1]:
			dict1.update({'Team_1': matches[j][0], 'Team_2': matches[j][1]})
		else:
			dict1.update({'Team_1': matches[j][1], 'Team_2': matches[j][0]})

#        # Append updated dictionary to the list, that will later be converted into a DataFrame
		pred_set.append(dict1)
		i += 2
		j += 1
        
#        # Convert list into DataFrame
	pred_set = pd.DataFrame(pred_set)
	backup_pred_set = pred_set

#    # Get dummy variables and drop winning_team column
	pred_set = pd.get_dummies(pred_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])

#    # Add missing columns compared to the model's training dataset
	missing_cols2 = set(final.columns) - set(pred_set.columns)
	for c in missing_cols2:
		pred_set[c] = 0
	pred_set = pred_set[final.columns]

	pred_set = pred_set.drop(['Winner'], axis=1)

#    # Predict!
	predictions = logreg.predict(pred_set)
	for i in range(len(pred_set)):
		print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
		if predictions[i] == 1:
			print("Winner: " + backup_pred_set.iloc[i, 1])
		else:
			print("Winner: " + backup_pred_set.iloc[i, 0])
			print("")



def clean_and_predict():
	clean_and_predict1(semi, ranking, final, rf)



l9 = Label(root,text="clean and predict semi finals result",font='Helvetica 14 bold',fg='tomato').grid(row=48,column=0)

clean_pred_button=Button(root,text="predict",width=15,activebackground="green",command=clean_and_predict).grid(row=48,column=2,padx=40,pady=0)


############################################################################################################################################

def clean_and_predict_final():
	global finals
	finals = [('India', 'England')]
	print("\n\n")

	clean_and_predict1(finals, ranking, final, rf)

	messagebox.showinfo("Final Result","Final winner is : India")
#	final_res = Message(root, text="Winner of final match is : India", width=50)
#	final_res.pack()

l10 = Label(root,text="clean and predict final result",font='Helvetica 14 bold',fg='tomato').grid(row=52,column=0)

final_pred_button=Button(root,text="predict",width=15,activebackground="green",command=clean_and_predict_final).grid(row=52,column=2,padx=40,pady=0)


############################################################################################################################################


root.mainloop()
