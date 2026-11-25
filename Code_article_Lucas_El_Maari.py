import numpy as np
import matplotlib.pyplot as plt
import time

# ========================================================================
# SECTION 1 : PARAMÈTRES PHYSIQUES DU PROPULSEUR SPT-100
# ========================================================================

class ParametresPropulseur:
    L = 0.04    #Longueur du canal en m
    Nx = 200    #Point de grille
    qe = 1.6e-19    #Charge élémentaire en C
    eps0 = 8.85e-12 #Permittivité du vide en F/M
    mi = 2.18e-25   #Masse d'un ion Xe+ kg   
    me = 9.11e-31   #Masse électron en kg
    Va = 300    #Tension anode en V
    Te = 15 * qe    #Température électronique en J ~15eV
    x0 = 0.022      # Centre de la zone d'ionisation (m)
    w = 0.006       # Largeur de la zone (m)
    S0 = 5e22       # Intensité de la source (m⁻³s⁻¹)
    def __init__(self):
        self.x = np.linspace(0, self.L, self.Nx)
        self.dx = self.x[1] - self.x[0]

# ========================================================================
# SECTION 2 : SOURCE D'IONISATION
# ========================================================================

def source_ionisation(params):
    return params.S0 * np.exp(-((params.x - params.x0)/params.w)**2)

# ========================================================================
# SECTION 3 : MODÈLE FLUIDE
# ========================================================================

def modele_fluide(params):
    #initialisation
    n = source_ionisation(params).copy()
    u = np.zeros(params.Nx)
    phi = np.linspace(params.Va, 0, params.Nx)
    dt = 5e-9 # pas de temps
    n_iterations = 1500
    for _ in range(n_iterations):
        #Loi de Botltzmann
        phi_ref = np.max(phi)
        ne = n*np.exp(params.qe*(phi-phi_ref)/params.Te)
        ne = np.maximum(ne, 1e14)
        
        #Equation de Poisson
        rho = params.qe*(n-ne)
        phi[0], phi[-1] = params.Va, 0
        for _ in range(40):
            for i in range(1, params.Nx-1):
                phi[i] = 0.5*(phi[i-1]+phi[i+1]+params.dx**2*rho[i]/params.eps0)
        phi = np.clip(phi, 0, params.Va)
        E = -np.gradient(phi, params.dx) #Champ électrique
        u += params.qe*E/params.mi*dt #Quantité de mouvement
        
        #Equations de continuité
        n -= dt*np.gradient(n*u, params.dx)
        n += source_ionisation(params)*dt
        n = np.maximum(n, 1e14)
    return n, phi

# ========================================================================
# SECTION 4 : MODÈLE HYBRIDE
# ========================================================================

def modele_hybride(params):
    
    #Initialisation
    Np = 12000
    xp = np.random.uniform(0, params.L/4, Np)
    vp = np.zeros(Np)
    phi = np.linspace(params.Va, 0, params.Nx)
    dt = 1e-9 #Pas de temps
    alpha = 0.6 # Facteur de sous-relaxation
    n_iterations = 1500
    for _ in range(n_iterations):
        
        #Dépôt de charge (Cloud-in-Cell)
        ni = np.zeros(params.Nx) 
        for i in range(Np):
            pos = xp[i]/params.dx
            i0=int(np.floor(pos))
            i1=i0+1
            w1=pos-i0
            w0=1-w1
            if 0<=i0<params.Nx: ni[i0]+=w0
            if 0<=i1<params.Nx: ni[i1]+=w1
        ni = ni/params.dx + source_ionisation(params)*dt #Ajout de la source
        ni = np.convolve(ni, np.ones(5)/5, mode='same')
        
        #Loi de Boltzmann
        phi_ref = np.max(phi)
        ne = ni*np.exp(params.qe*(phi-phi_ref)/params.Te)
        ne = np.maximum(ne, 1e14)
        
        #Equation de Poisson avec le facteur alpha
        rho = params.qe*(ni-ne)
        phi_new = phi.copy()
        phi_new[0], phi_new[-1] = params.Va, 0
        for _ in range(60):
            for i in range(1, params.Nx-1):
                phi_new[i]=0.5*(phi_new[i-1]+phi_new[i+1]+params.dx**2*rho[i]/params.eps0)
        phi = alpha*phi_new+(1-alpha)*phi
        phi = np.convolve(phi,np.ones(5)/5,mode='same')
        phi = np.clip(phi,0,params.Va)
        E = -np.gradient(phi, params.dx) #Champ électrique
        
        #Avancement des particules
        for i in range(Np):
            idx = int(xp[i]/params.dx)
            if 0<=idx<params.Nx: vp[i]+=params.qe*E[idx]/params.mi*dt
            xp[i]+=vp[i]*dt
    return ni, phi

# ========================================================================
# SECTION 5 : MODÈLE PIC COMPLET
# ========================================================================

def modele_pic(params):
    
    #Initialisation
    Np = 20000
    xp = np.random.rand(Np)*params.L
    vp = np.random.randn(Np)*300
    
    # Attribution charge et masse
    charge = np.where(np.arange(Np)<Np//2,-params.qe,params.qe)
    mass = np.where(np.arange(Np)<Np//2,params.me,params.mi)
    phi = np.zeros(params.Nx)
    dt = 2e-11 #Pas de temps
    n_iterations = 1500
    for _ in range(n_iterations):
        
        # Dépot de charge CIC
        rho = np.zeros(params.Nx)
        ni = np.zeros(params.Nx)
        for i in range(Np):
            pos = xp[i]/params.dx
            i0=int(np.floor(pos))
            i1=i0+1
            w1=pos-i0
            w0=1-w1
            if 0<=i0<params.Nx: 
                rho[i0]+=charge[i]*w0
                if charge[i]>0: ni[i0]+=w0
            if 0<=i1<params.Nx:
                rho[i1]+=charge[i]*w1
                if charge[i]>0: ni[i1]+=w1
        ni=ni/params.dx
        rho=rho/params.dx
        
        #Equation de Poisson
        phi[0], phi[-1]=params.Va,0
        for _ in range(50):
            for i in range(1, params.Nx-1):
                phi[i]=0.5*(phi[i-1]+phi[i+1]+params.dx**2*rho[i]/params.eps0)
        E=-np.gradient(phi, params.dx)  #Champ électrique
        
        #Avancement des particules
        xp += vp*dt
        for i in range(Np):
            idx=int(xp[i]/params.dx)
            if 0<=idx<params.Nx: vp[i]+=charge[i]*E[idx]/mass[i]*dt
        
        #Conditions aux limites
        out_left = xp<0
        out_right=xp>params.L
        xp[out_left]=0; vp[out_left]=0
        xp[out_right]=params.L; vp[out_right]=0
    ni = np.convolve(ni,np.ones(7)/7,mode='same')
    return ni, phi

# ========================================================================
# SECTION 6 : PROFIL EXPÉRIMENTAL
# ========================================================================

def profil_experimental(params):
    phi_exp = params.Va*(1-np.tanh((params.x-0.022)/0.007))/2
    return np.clip(phi_exp,0,params.Va)

# ========================================================================
# SECTION 7 : METRIQUES
# ========================================================================

def calculer_dphi(phi_num, phi_exp):
    return np.sqrt(np.mean((phi_num-phi_exp)**2))

# ========================================================================
# SECTION 8 : VISUALISATIONS PAR MODÈLE 
# ========================================================================

def visualisation_par_modele(params, resultats, phi_exp):
    x_mm = params.x*1e3
    models = ['Fluide','Hybride','PIC']
    densites = [resultats['fluide'][0], resultats['hybride'][0], resultats['pic'][0]]
    potentiels = [resultats['fluide'][1], resultats['hybride'][1], resultats['pic'][1]]
    colors = ['blue','orange','green']

    # Densité ionique
    plt.figure(figsize=(8,12))
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(x_mm,densites[i]/np.max(densites[i]),color=colors[i],linewidth=2)
        plt.fill_betweenx([0,1],(params.x0-params.w)*1e3,(params.x0+params.w)*1e3,color='gray',alpha=0.2)
        plt.title(f"Densité ionique - {models[i]}")
        plt.xlabel("Position (mm)")
        plt.ylabel("Densité normalisée")
        plt.grid(True)
    plt.tight_layout()
    plt.savefig("densite_par_modele.png",dpi=150)
    plt.show()

    # Potentiel électrique avec erreur
    plt.figure(figsize=(8,12))
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(x_mm, phi_exp, 'k', label='Expérience')
        plt.plot(x_mm, potentiels[i], color=colors[i], linestyle='--', label=models[i])
        
        # Affichage de l'erreur
        plt.fill_between(x_mm, phi_exp, potentiels[i], color=colors[i], alpha=0.2)
        plt.text(0.98, 0.95, f'dphi = {calculer_dphi(potentiels[i], phi_exp):.2f} V',
                 transform=plt.gca().transAxes, ha='right', va='top',
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title(f"Potentiel électrique - {models[i]}")
        plt.xlabel("Position (mm)")
        plt.ylabel("Potentiel (V)")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig("potentiel_par_modele.png",dpi=150)
    plt.show()


def visualisation_dphi_temps(temps, dphi_values):
    models=['Fluide','Hybride','PIC']
    colors=['skyblue','orange','lightgreen']
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    bars = plt.bar(models, temps, color=colors, edgecolor='black')
    plt.ylabel("Temps de calcul (s)", fontsize = 18)
    plt.title("Coût computationnel", fontsize = 18)
    for i, b in enumerate(bars): plt.text(b.get_x()+b.get_width()/2, b.get_height()*1.01,f"{temps[i]:.3f}",ha='center', fontsize=14)
    plt.subplot(1,2,2)
    bars = plt.bar(models, dphi_values, color=colors, edgecolor='black')
    plt.ylabel("dphi (V)", fontsize = 18)
    plt.title("Précision vs expérience", fontsize = 18)
    for i, b in enumerate(bars): plt.text(b.get_x()+b.get_width()/2, b.get_height()*1.01,f"{dphi_values[i]:.1f}",ha='center', fontsize=14)
    plt.tight_layout()
    plt.savefig("dphi_temps_comparaison.png",dpi=150)
    plt.show()


# ========================================================================
# PROGRAMME PRINCIPAL
# ========================================================================

def main():
    params = ParametresPropulseur()
    phi_exp = profil_experimental(params)
    resultats = {}

    # Calcul des modèles
    start = time.time()
    nf, phif = modele_fluide(params)
    tf = time.time() - start
    resultats['fluide'] = (nf, phif)

    start = time.time()
    nh, phih = modele_hybride(params)
    th = time.time() - start
    resultats['hybride'] = (nh, phih)

    start = time.time()
    npic, phip = modele_pic(params)
    tpic = time.time() - start
    resultats['pic'] = (npic, phip)

    # Calcul dphi
    dphi_values = [
        calculer_dphi(phif, phi_exp),
        calculer_dphi(phih, phi_exp),
        calculer_dphi(phip, phi_exp)
    ]
    temps = [tf, th, tpic]
    print("dphi (V) par modèle : Fluide, Hybride, PIC =", [f"{r:.2f}" for r in dphi_values])
    print("Temps de calcul (s) : Fluide, Hybride, PIC =", [f"{t:.3f}" for t in temps])

    # Visualisations
    visualisation_par_modele(params, resultats, phi_exp)
    visualisation_dphi_temps(temps, dphi_values)


# ========================================================================
# EXECUTION
# ========================================================================

if __name__=="__main__":
    main()