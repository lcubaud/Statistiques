
import numpy as np
from random import *
import matplotlib.pyplot as plt
import time

""" On va considérer que la position d'un bateau est sa case la plus à gauche s'il est
horizontal, et sa case la plus en haut dans le cas où il est vertical. On va aussi faire l'hypoyhèse
que comme dans loa bataille navale on ne colle pas les bateaux et on ne connait pas le type de bateau
que l'on touche. """

dict = { 0 : ['vide', 0], 1 : ['porte-avions', 5], 2 : ['croiseur', 4], 3 : ['contre-torpilleurs', 3], 4 : ['sous-marin', 3], 5 : ['torpilleur', 2] }

                                            ### Partie N°1 : Modélisation et fonctions simples

def peut_placer(grille, bateau, position, direction) :

    """ list[list[int]] * int * tuple[int, int] * int -> bool
    Cette fonction retourne vrai s’il est possible de placer le bateau sur la grille.
    """
    # v : int
    v = 0
    # h : int
    h = 0
    # (x,y) : tuple[int,int]
    (x,y) = position
    # i : int
    for i in range((dict[bateau])[1]) :
        if direction == 1 :
            h = i
        else :
            v = i
        if x+v >= (len(grille)) or y+h >= (len(grille[0])) or grille[x+v][y+h] != 0 :
            return False
    if bateau_non_colle(grille, bateau, position, direction):
        return True
    else:
        return False
   
        
def bateau_non_colle(grille, bateau, position, direction):
    
    """ list[list[int]] * int * tuple[int, int] * int -> bool"""
    (x,y) = position
    if direction==1:       
        if x+1 >= len(grille):
            for i in range((dict[bateau])[1]):
                if grille[x-1][y+i] > 0:
                    return False
        elif x-1 <0 :
            for i in range ((dict[bateau])[1]):
                if grille[x+1][y+i] > 0:
                    return False
        else:
            for i in range ((dict[bateau])[1]):
                if grille[x+1][y+i] > 0 or grille[x-1][y+i] > 0:
                    return False
        if y-1<0:
            if grille[x][y+(dict[bateau])[1]] > 0:
                return False
        elif y+(dict[bateau])[1] >= len(grille):
            if  grille[x][y-1] > 0:
                return False
        else:
            if grille[x][y+(dict[bateau])[1]] > 0 or grille[x][y-1] > 0:
                return False
    else:
        if y+1 >= len(grille):
            for i in range((dict[bateau])[1]):
                if grille[x+i][y-1] > 0:
                    return False
        elif y-1 <0 :
            for i in range ((dict[bateau])[1]):
                if grille[x+i][y+1] > 0:
                    return False
        else:
            for i in range ((dict[bateau])[1]):
                if grille[x+i][y+1] > 0 or grille[x+i][y-1] > 0:
                    return False
             
        if x-1<0:
            if grille[x+(dict[bateau])[1]][y] > 0:
                return False
        elif x+(dict[bateau])[1] >= len(grille):
            if  grille[x-1][y] > 0:
                return False
        else:
            if grille[x+(dict[bateau])[1]][y] > 0 or grille[x-1][y] > 0:
                return False
    return True

def place(grille, bateau, position, direction) :

    """ list[list[int]] * int * tuple[int, int] * int -> bool
    Cette fonction retourne la grille modifiée s’il est possible de placer le bateau.
    """
    # v : int
    v = 0
    # h : int
    h = 0
    # (x,y) : tuple[int,int]
    (x,y) = position
    # i : int
    for i in range((dict[bateau])[1]) :
        if direction == 1 :
            h = i
        else :
            v = i
        grille[x+v][y+h] = bateau
    return grille

def place_alea(grille, bateau) :

    """ list[list[int]] * int  -> None
    Place aléatoirement le bateau dansla grille : la fonction tire uniformément
    une position et une direction aléatoires et tente de placer le bateau ; s’il
    n’est pas possible de placer le bateau, un nouveau tirage est effectué et ce
    jusqu’à ce que le positionnement soit admissible.
    """
    # (x,y) : tuple[int,int]
    (x,y) = (randint(0, 9),randint(0, 9))
    # direction : int
    direction = randint(1,2)

    while not (peut_placer(grille, bateau, (x,y), direction)) :
        (x,y) = (randint(0, 9),randint(0, 9))
        direction = randint(1,2)
    place(grille, bateau, (x,y), direction)

def affiche(grille) :

    """ list[list[int]] * int  -> None
    Affiche la grille de jeu (utiliser imshow du module matplotlib.pyplot).
    """
    plt.grid(True)
    plt.imshow(grille,origin="lower").set_cmap("jet")
    plt.title('Bataille Navale', fontsize=15)
    plt.show()

def eq(grilleA,grilleB) :

    """ list[list[int]] * list[list[int]] -> bool
    Tester l’égalité entre deux grilles
    """
    # i : int
    for i in range (len(grilleA)):
        # j : int
        for j in range (len(grilleA[0])):
            if grilleA[i][j] != grilleB[i][j]:
                return False
    return True

def genere_grille():

    """ None -> list[list[int]]
    Rend une grille avec les bateaux disposés de manière aléatoire
    """
    # grille : ndarray
    grille = np.zeros((10,10),dtype=int)
    # i : int
    for i in range (1, 6):
        place_alea(grille, i)
    return grille

                                            ### Partie N°2 : Combinatoire du jeu

def placer_un_bateau(bateau) :

    """ int -> int
    Retourne le nombre de possibilité de placer un bateau donné sur une grille vide.
    """
    # grille : ndarray
    grille = np.zeros((10,10),dtype=int)
    # cpt : int
    cpt = 0
    # i : int
    for i in range(10):
        # j : int
        for j in range(10):
            if (peut_placer(grille, bateau, (i,j), 1)):
                cpt=cpt+1
            if (peut_placer(grille, bateau, (i,j), 2)):
                cpt=cpt+1
    return cpt


def placer_des_bateaux(grille,liste_bateau) :

    """ list[list[int]] * list[int] -> int
    Hypoyhèse : liste_bateau_bateau contient au moins un bateau
    Retourne le nombre de possibilité de placer une liste_bateau de bateaux sur une grille vide.
    """
    # cpt : int
    cpt = 0
    if len(liste_bateau) == 0 :
        return 1
    # i : int
    for i in range(10):
        # j : int
        for j in range(10):
            if (peut_placer(grille, liste_bateau[0], (i,j), 1)):
                cpt += placer_des_bateaux(place(grille.copy(), liste_bateau[0], (i,j), 1), liste_bateau[1:])
            if (peut_placer(grille, liste_bateau[0], (i,j), 2)):
                cpt += placer_des_bateaux(place(grille.copy(), liste_bateau[0], (i,j), 2), liste_bateau[1:])
    return cpt

def genere_grille_listebateau(liste_bateau):

    """ list[int] -> list[list[int]]
    Retourne une grille avec les bateaux de liste_bateau disposés de manière aléatoire.
    """
    # grille : ndarray
    grille = np.zeros((10,10),dtype=int)
    # i : int
    for i in liste_bateau:
        place_alea(grille, i)
    return grille

def grille_egale_alea(grille, liste_bateau):

    """ list[list[int]] * list(int) -> int
    Prend en paramètre une grille, génère des grilles aléatoirement jusqu’à ce
    que la grille générée soit égale à la grille passée en paramètre et
    renvoie le nombre de grilles générées.
    """
    # grillegeneree : ndarray
    grillegeneree = genere_grille_listebateau(liste_bateau)
    # cpt : int
    cpt = 1
    while not eq(grille, grillegeneree) :
        grillegeneree = genere_grille_listebateau(liste_bateau)
        cpt += 1
    return cpt

def test_grille_egale_alea(duree, liste_bateau) :

    """ int * list[int] -> int
     Pour une durée "duree" donnée (en seconde), on itère autant que possible sur la fonction
    grille_egale_alea pour donner au final une moyenne de nombre de grilles générées.
    """
    print()
    print("Test grille_egale_alea ("+str(duree)+"s, "+ str(liste_bateau) +")")
    cpt = 0
    nb = 1
    res = 0
    start_time = time.process_time()
    end_time = 0
    while (True) :
        end_time = time.process_time()
        if (end_time - start_time > duree) :
            break
        cpt += grille_egale_alea(genere_grille_listebateau(liste_bateau), liste_bateau)
        nb += 1
    res = (cpt // nb)
    print("Moyenne pour grille_egale_alea avec "+ str(liste_bateau) +" : " + str(res))
    print("(" + str(nb) + " itérations)")
    print("("+ str(round(end_time - start_time)) +"s)")
    return res

def approx_nb_grilles1(liste_bateau) :
    """ list[int] -> int
    Retourne une approximation du nombre de grilles différentes possibles
    contenant la liste de bateaux passée en paramètre. On multiplie les
    résultats respectifs de la fonction placer_un_bateau(...) appliquée aux
    différents bateaux de liste_bateau.
    """
    # res : int
    res = 1
    # i : int
    for i in liste_bateau :
        res *= placer_un_bateau(i)
    return res
    
##Question Bonus

def placer_un_bateau2(grille,bateau) :

    """ list[list[int]] * int -> int
    Retourne le nombre de possibilité de placer un bateau donné sur une grille donnée.
    """
    # cpt : int
    cpt = 0
    # i : int
    for i in range(10):
        # j : int
        for j in range(10):
            if (peut_placer(grille, bateau, (i,j), 1)):
                cpt=cpt+1
            if (peut_placer(grille, bateau, (i,j), 2)):
                cpt=cpt+1
    return cpt


def approx_nb_grilles2(liste_bateau) :
    """ list[int] -> int
    Retourne une approximation du nombre de grilles différentes possibles
    contenant la liste de bateaux passée en paramètre.
    """
    # res : int
    res = 1
    # grille : ndarray
    grille = np.zeros((10,10),dtype=int)
    for i in liste_bateau :
        res *= placer_un_bateau2(grille,i)
        place_alea(grille, i)
    return res
    
    
def trouveMax(grille):
    """ list[list[int]] -> tuple[int,int] """
    # max : int 
    max=0
    # (xMax,yMax) : tuple[int, int]
    (xMax,yMax)=(0,0)
    for x in range(10):
        for y in range(10):
            if(grille[x][y]<1):
                if((grille[x][y])>max):
                    max=grille[x][y]
                    (xMax,yMax)=(x,y)
    return (xMax,yMax)

                                            ### Partie N°3 : Modélisation probabiliste du jeu

class Bataille:

    def __init__(self):
        self.tab=[1,1,1,1,1,2,2,2,2,3,3,3,4,4,4,5,5]
        self.tabDebut=self.tab.copy()
        self.grille = genere_grille()
        self.grilleDebut = self.grille.copy()


    def joue(self,position):
        # (x,y) : tuple[int, int]
        (x,y) = position
        if(self.grille[x][y]==0):
            return ("raté")
        else:
            for i in range(len(self.tab)):
                if self.tab[i] == self.grille[x][y]:
                    self.tab[i] = 0
                    break;
            for i in range(len(self.tab)):
                if self.tab[i] == self.grille[x][y]:
                    return ("touché")
            return ("coulé")

    def victoire(self):
        for i in self.tab:
            if i!=0:
                return False
        return True


    def reset(self):
        self.grille=self.grilleDebut
        self.tab=self.tabDebut

class Joueur:
    
   #Version Aléatoire

    def JoueurAleatoire(self):
        cpt = 0
        b = Bataille()
        grille_choix = np.zeros((10,10),dtype=int)
        #print(b.grille)
        while not (b.victoire()):
            (x,y) = (randint(0, 9),randint(0, 9))
            if(grille_choix[x][y]==0):
                grille_choix[x][y]=-1
                b.joue((x,y))
                cpt += 1
        return cpt
    
    #Version Heuristique

    def JoueurHeuristique(self):
        cpt = 0
        b = Bataille()
        grille_choix = np.zeros((10,10),dtype=int)
        parcours_tab =[(0,1),(0,-1),(1,0),(-1,0)]
        
        while not (b.victoire()):
            #on cherche de manière aléatoire
            (x,y) = (randint(0, 9),randint(0, 9))        
            if(grille_choix[x][y]==0):
                grille_choix[x][y]=-1
                res = b.joue((x,y))
                #des qu'on touche un bateau, on va chercher à le couler
                if (res=="touché"):
                    trouve=False
                    cptTest=0
                    #on parcours les cases adjacentes
                    for i in range(4):
                        if(not trouve):
                            (xTest,yTest)=(x+parcours_tab[i][0],y+parcours_tab[i][1])
                            if((xTest<=9 and yTest<=9 and xTest>=0 and yTest>=0) and grille_choix[xTest][yTest]==0):
                                res=b.joue((xTest,yTest))
                                grille_choix[xTest][yTest]=-1
                                cptTest+=1
                                if(res=="coulé"):
                                    trouve=True
                                    cpt+=cptTest
                                
                                if(res=="touché"):
                                    trouve=True
                                    cpt+=cptTest
                                    (xInit,yInit)=(x,y)
                                    (x,y)=(xTest,yTest)
                                    while(res!="coulé"):
                                        #Il reste une partie du bateau à toucher donc on va dans
                                        #l'autre sens
                                        if(res=="raté"):
                                            (x,y)=(xInit,yInit)
                                            if(i%2==0):
                                                i+=1
                                            else:
                                                i-=1
                                        (x,y)= (x+parcours_tab[i][0],y+parcours_tab[i][1])
                                        if(x<=9 and y<=9 and x>=0 and y>=0):
                                            grille_choix[x][y]=-1
                                            res=b.joue((x,y))
                                            cpt+=1
                                        else:
                                            res="raté"
                cpt += 1
        return cpt


    #Version Probabiliste Simplifiee

    def JoueurProbabilisteSimplifiee(self):
        cpt=0
        b = Bataille()
        bateauATrouver=[1,2,3,4,5]
        grille_proba = [[10] * 10 for _ in range(10)]
        grille_bonus=np.zeros((10,10),dtype=float)
        grille_proba=self.calcul(grille_proba,bateauATrouver,grille_bonus)
        while not(b.victoire()):
            (x,y)=trouveMax(grille_proba)
            res=b.joue((x,y))
            cpt+=1
            if(res=="raté"):
                grille_proba[x][y]=0
                grille_proba=self.calcul(grille_proba,bateauATrouver,grille_bonus)
            if(res=="touché"):
                grille_proba[x][y]=1
                grille_proba=self.calcul(grille_proba,bateauATrouver,grille_bonus)
            if(res=="coulé"):
                grille_proba[x][y]=1
                bateau=self.trouve(x,y,grille_proba)
                if(bateau not in bateauATrouver):
                    bateau=4
                bateauATrouver.remove(bateau)
        return cpt
    
    
    def calcul(self,grille_proba,bateauATrouver,grille_bonus):
        #initialise grille_choix
        grille=np.zeros((10,10),dtype=float)
        grille_choix=np.zeros((10,10),dtype=float)
        for x in range (10):
            for y in range (10):
                if grille_proba[x][y]==0 or grille_proba[x][y]==-1:
                    grille_choix[x][y]=-1
                else:
                    grille_choix[x][y]=0
        #initialise grille_proba
        for i in range(len(bateauATrouver)):
                for x in range(10):
                    for y in range(10):
                        if(peut_placer(grille_choix,bateauATrouver[i],(x,y),1)):
                            if(grille_proba[x][y]!=0):
                                for t in range (dict[bateauATrouver[i]][1]):
                                       grille[x][y+t]+=1
                        if(peut_placer(grille_choix,bateauATrouver[i],(x,y),2)):
                            if(grille_proba[x][y]!=0):
                                for t in range (dict[bateauATrouver[i]][1]):
                                    if(grille_proba[x+t][y]!=1):
                                       grille[x+t][y]+=1
        #on replace les bateaux déjà trouvés
        for x in range(10):
            for y in range(10):
                if grille_proba[x][y]==1:
                    grille[x][y]=1
                if grille_proba[x][y]==-1:
                    grille[x][y]=-1
        
        for i in range(10):
            for j in range(10):
                if(grille_proba[i][j]!=1 and grille_proba[i][j]!=-1):
                    grille[i][j]=grille[i][j]/100
        
        #ajoute un bonus si bateau trouvé dans les cases adjacentes
        for x in range(10):
            for y in range(10):
                if self.one_around(x,y,grille):
                    if grille[x][y]!=0 and grille[x][y]!=1 and grille_proba[i][j]!=-1:
                        grille_bonus[x][y]+=0.1
                        grille[x][y]+=grille_bonus[x][y]
        return grille
    
    def one_around(self,x,y,grille_proba):
        parcours_tab =[(0,1),(0,-1),(1,0),(-1,0)]
        for i in range(4):
            (xTest,yTest)=(x+parcours_tab[i][0],y+parcours_tab[i][1])
            if((xTest<=9 and yTest<=9 and xTest>=0 and yTest>=0)):
                if grille_proba[xTest][yTest]==1:
                    return True
        return False
    
    
    def trouve(self,x,y,grille_proba):
        taille=0
        parcours_tab =[(0,1),(0,-1),(1,0),(-1,0)]
        taille+=1
        grille_proba[x][y]=-1
        for i in range(4):
            (xTest,yTest)=(x+parcours_tab[i][0],y+parcours_tab[i][1])
            if((xTest<=9 and yTest<=9 and xTest>=0 and yTest>=0)):
                if grille_proba[xTest][yTest]==1:
                    (xI,yI)=(xTest,yTest)
                    while(grille_proba[xI][yI]==1 ):
                        grille_proba[xI][yI]=-1
                        (xI,yI)=(xI+parcours_tab[i][0],yI+parcours_tab[i][1])
                        taille+=1
                        if not (xI<=9 and xI>=0 and yI<=9 and yI>=0):
                            break
        for i in range (6):
            if taille == dict[i][1]:
                return i
        
    #Version Bonus

    def JoueurBonus(self):
            cpt=0
            b = Bataille()
            bateauATrouver=[1,2,3,4,5]
            grille_choix = np.zeros((10,10),dtype=int)
            
            grille_bateau=self.calculBonus(grille_choix,bateauATrouver)          
                 
            while not(b.victoire()):
                (xM,yM)=self.myMax(grille_bateau)
                if(grille_choix[xM][yM]==0):
                    if(b.joue((xM,yM))=="raté"):
                        grille_bateau[xM][yM]=-1
                        grille_choix[xM][yM]=-1
                        cpt+=1
                    else:
                        grille_bateau[xM][yM]=0
                        grille_choix[xM][yM]=-1
                        cpt+=1
                        #on cherche le bateau de la meme maniere que v2
                        (grille_choix,cpt,bateau)=self.heuristique(xM,yM,grille_choix,b,cpt)
                        if(bateau > 0):
                            if(bateau not in bateauATrouver):
                                bateau=4
                            bateauATrouver.remove(bateau)
                            bateau=0                       
                    grille_bateau=self.calculBonus(grille_choix,bateauATrouver)
    
            return cpt
    
            
    def calculBonus(self,grille_choix,bateauATrouver):
        grille=np.zeros((10,10),dtype=float)
        for i in range(len(bateauATrouver)):
                for x in range(10):
                    for y in range(10):
                        if(peut_placer(grille_choix,bateauATrouver[i],(x,y),1)):
                            for t in range (dict[bateauATrouver[i]][1]):
                                grille[x][y+t]+=1
                        if(peut_placer(grille_choix,bateauATrouver[i],(x,y),2)):
                            for t in range (dict[bateauATrouver[i]][1]):
                                grille[x+t][y]+=1
        
        return grille
    
    def heuristique(self,x,y,grille_choix,b,cpt):
        parcours_tab =[(0,1),(0,-1),(1,0),(-1,0)]
        cptTest=0
        tailleBateau=1
        trouve=False
        for i in range(4):
            if not (trouve):
                (xTest,yTest)=(x+parcours_tab[i][0],y+parcours_tab[i][1])
                if((xTest<=9 and yTest<=9 and xTest>=0 and yTest>=0) and grille_choix[xTest][yTest]==0):
                    res=b.joue((xTest,yTest))
                    grille_choix[xTest][yTest]=-1
                    cpt+=1
                    if(res=="coulé"):
                        trouve=True
                    if(res=="touché" ):               
                        trouve=True
                        (xInit,yInit)=(x,y)
                        (x,y)=(xTest,yTest)
                        
                        while(res!="coulé"):
                            if(res=="raté"):
                                (x,y)=(xInit,yInit)
                                if(i%2==0):
                                    i+=1
                                else:
                                    i-=1
                            if(res=="touché"):
                                tailleBateau+=1
                            x=x+parcours_tab[i][0]
                            y=y+parcours_tab[i][1]
                            if(x<=9 and y<=9 and x>=0 and y>=0):
                                grille_choix[x][y]=-1
                                res=b.joue((x,y))
                                cpt+=1                              
                            else:
                                res="raté"                                                     
                        break
        
        tailleBateau+=1
        for i in range (5):
            if tailleBateau == dict[i][1]:
                bateau=i
                return (grille_choix,cpt,bateau)
        return (grille_choix,cpt,-1)


    def myMax(self,grille):
        max=0
        (xMax,yMax)=(0,0)
        for x in range(10):
            for y in range(10):
                if((grille[x][y])>max):
                    max=grille[x][y]
                    (xMax,yMax)=(x,y)
        return (xMax,yMax)   



### Jeu de tests

grille_test = np.zeros((10,10),dtype=int)
assert peut_placer(grille_test, 1, (0, 0), 1)== True
assert peut_placer(grille_test, 1, (9, 9), 1) == False

assert peut_placer(grille_test, 1, (9, 0), 1) == True
assert peut_placer(grille_test, 2, (9, 0), 1) == True
assert peut_placer(grille_test, 1, (0, 9), 1) == False
assert peut_placer(grille_test, 2, (0, 9), 1) == False


affiche(genere_grille())

print("Test placer_un_bateau [3]")
print(placer_un_bateau(3))

print("Test placer_des_bateaux [5]")
print(placer_des_bateaux(grille_test,[5]))

print("Test placer_des_bateaux [5,4]")
print(placer_des_bateaux(grille_test,[5,4]))



print("           Test placer_des_bateaux   ", "approx_nb_grilles1 ","approx_nb_grilles2")
print("[5]            ", str(placer_des_bateaux(grille_test,[5]))+"                ", "       "+str(approx_nb_grilles1([5]))+"                "+str(approx_nb_grilles2([5])))
print("[5, 4]         ", str(placer_des_bateaux(grille_test,[5,4]))+"              ", "       "+str(approx_nb_grilles1([5,4]))+"             "+str(approx_nb_grilles2([5,4])))
#print("[5, 4, 3]      ", str(placer_des_bateaux(grille_test,[5,4,3]))+"            ", "       "+str(approx_nb_grilles1([5,4,3]))+"             "+str(approx_nb_grilles2([5,4,3])))



#print(test_grille_egale_alea(60,[5]))


#   Test Version Aleatoire
"""
tab = np.zeros((101),dtype=int)
cpt=0
for i in range (100):
    j=Joueur()
    res=j.JoueurAleatoire()
    tab[res]+=1
    cpt+=res
print("Joueur aleatoire -- nombre moyen coup : ",cpt/100)
plt.title("Version aléatoire -- graphe de distribution")
plt.plot(tab)
plt.xlabel('Nombre de coups')
plt.ylabel('Distribution')
plt.show()
"""
#   Test Version Heuristique
"""
tab = np.zeros((101),dtype=int)
cpt=0
for i in range (100):
    j=Joueur()
    res=j.JoueurHeuristique()
    tab[res]+=1
    cpt+=res
print("Joueur heuristique -- nombre moyen coup : ",cpt/100)
plt.title("Version heuristique -- graphe de distribution")
plt.plot(tab)
plt.xlabel('Nombre de coups')
plt.ylabel('Distribution')
plt.show()
"""
#   Test Version Probabiliste Simplifiee
"""
tab = np.zeros((101),dtype=int)
cpt=0
for i in range (100):
    j=Joueur()
    res=j.JoueurProbabilisteSimplifiee()
    tab[res]+=1
    cpt+=res
print("Joueur probabiliste simplifiee -- nombre moyen coup : ",cpt/100)
plt.title("Version probabiliste simplifiee -- graphe de distribution")
plt.plot(tab)
plt.xlabel('Nombre de coups')
plt.ylabel('Distribution')
plt.show() 
"""
#   Test Bonus
"""
tab = np.zeros((101),dtype=int)
cpt=0
for i in range (100):
    j=Joueur()
    res=j.JoueurBonus()a
    tab[res]+=1
    cpt+=res
print("Joueur Bonus -- nombre moyen coup : ",cpt/100)
plt.title("Version Bonus -- graphe de distribution")
plt.plot(tab)
plt.xlabel('Nombre de coups')
plt.ylabel('Distribution')
plt.show() 

"""