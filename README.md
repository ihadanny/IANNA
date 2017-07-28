Hi Ido, I opened a new git repo and invited you.
It is called IANNA (which is Interactive Analysis NN Agent). 
Do you like the name? haha its also a Scottish female name ^__^

Also, I converted all datafiles to dataframes and pickled them, so you can skip the CSV loading..

Several followups from our meeting today!

    Can we use python3 and do classes and object oriented stuff?
    Also, regarding the actions we need to implement: GROUP-BY and aggregate; and (2) Filter on a single attribute
    Note that pandas does not trivially support FILTER/GROUP operations on a dataframe.groupby objects, yet in our setting we allow a further filtering / another groupby action, I hope its not too difficult 
    Last, I attach a CSV with the actions I showed you today, can we use the same actions' syntax?



Don't hesitate to call me whenever you want,
Amit


**PS** This is the mapping of the filter operators both for string and numericals.
INT_OPERATOR_MAP = {
    8: operator.eq,
    32: operator.gt,
    64: operator.ge,
    128: operator.lt,
    256: operator.le,
    512: operator.ne,


}
STR_OPERATOR_MAP = {
    16: "%{}%", #contains
    2: "{}%", #begin with
    4: "%{}" #end with
}


Task list:
1. Constructing the "game":

    actions (filter, groupby, back) 
    states (summary of results set -> entropy, #distinct,#rows)

2. Agent:

    NN
    *challenge-> how to cope with large # of actions' parameters




Small tasks:
1. Open GIT repo
2. Create CSVs of the data. 
3. Ido is the architect of the game
4. Amit will see how we can represent the large action space ??????


Later:
1. Reward
2. Training (maybe with user data)