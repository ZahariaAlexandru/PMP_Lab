from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

'''
-un modelat , un joc de carti un fel de joc de poker simplificat - mecanism pariuri
- maxim 3 ture
- configurari
- retea benziana
- stabilire variabilele si care sunt depedentele intre variabile, cum sunt influentate una de alta
- noi stabilim parametriii
- mai putin cartile : ex 1/5
- alegem noi parametrii cand vrem sa pariam cand nu in functie de carti
- juc 2 cont ce  a facut juc 1
- depdente : care sunt parametrii
- ca modelul facut ex 1
- var aleatoare: - c1 c2 cartile fiecarui player depedente intre ele :p1 : 1/5 p2: 1/4
                 - r1 - pariaza/asteapta (player 1) 
                 - r2 - pariaza/asteapta/retrage (player 2) depinde de c2 si r1
                 - r3 -                           (player 1) - se termina runda depinde de
                 -
'''

joc_carti = BayesianNetwork(
    [
        ("C1", "C2"),
        ("C1", "R1"),
        ("C2", "R2"),
        ("R1", "R2"),
        ("R2", "R3"),
        ("C1", "R3")
    ]
)
CPD_C1 = TabularCPD(variable='C1', variable_card=5, values=[[0.2], [0.2],[0.2], [0.2], [0.2]])
print(CPD_C1)

CPD_C2 = TabularCPD(variable='C2', variable_card=5, values=[[0, 0.25, 0.25, 0.25, 0.25], [0.25, 0, 0.25, 0.25, 0.25], [0.25, 0.25, 0, 0.25, 0.25], [0.25, 0.25, 0.25, 0, 0.25], [0.25, 0.25, 0.25, 0.25, 0]], evidence=['C1'], evidence_card=[5])
print(CPD_C2)

CPD_R1 = TabularCPD(variable='R1', variable_card=2, values = [[0, 0.3, 0.3, 0.4, 0.6], [0.6, 0.4, 0.3, 0.3, 0]] , evidence=['C1'], evidence_card=[5])
print(CPD_R1)

CPD_R2 = TabularCPD(variable='R2', variable_card=3, values = [[0.4,0.5,0.1],[0.4,0.6,0.0]], evidence=['C2','P1'], evidence_card=[5,3])
print(CPD_R2)

CPD_R3 = TabularCPD(variable='R3', variable_card=2, values = [[0.5],[0.5]] , evidence=['C1','P1'], evidence_card=[5,3])
print(CPD_R3)

joc_carti.add_cpds(CPD_C1, CPD_C2, CPD_R1, CPD_R2, CPD_R3)
joc_carti.check_model()
