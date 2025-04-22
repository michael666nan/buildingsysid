import numpy as np

def estimate_properties(ss):
    H = heat_transfer_coef(ss)
    gA = solar_aperture(ss)
    Tc = time_constants(ss)
    
    return H, gA, Tc


def heat_transfer_coef(ss):
    A = ss.A 
    B = ss.B
    C = ss.C
    I = np.eye(A.shape[0]) 

    T = C @ np.linalg.inv(I - A) @ (B[:,0]+B[:,2])
    H = 1/(T-1)
    print(f"Heat transfer coefficient: {H[0]:.2f} W/K")
    
    return H[0]


def solar_aperture(ss):
    A = ss.A 
    B = ss.B
    C = ss.C
    I = np.eye(A.shape[0]) 
    Th = C @ np.linalg.inv(I - A) @ B[:,2]
    Ts = C @ np.linalg.inv(I - A) @ B[:,1]
    gA = Ts/Th
    print(f"Solar Aperture: {gA[0]:.2f} m^2")
    
    return gA[0]


def time_constants(ss):
    A = ss.A 
    
    # Calculate eigenvalues of A
    eigenvalues = np.linalg.eigvals(A)
    
    # Compute time constants based on system type
    if ss.samplingTime >0:
        # For discrete-time systems: τ = -Ts/ln(λ)
        # Only consider stable modes (|λ| < 1)
        time_constants = []
        for eig in eigenvalues:
            if 0 < abs(eig) < 1:  # Stable discrete eigenvalue
                tau = -ss.samplingTime / np.log(abs(eig))
                time_constants.append(tau)
    else:
        # For continuous-time systems: τ = -1/λ
        # Only consider stable modes (Re(λ) < 0)
        time_constants = []
        for eig in eigenvalues:
            if np.real(eig) < 0:  # Stable continuous eigenvalue
                tau = -1 / np.real(eig)
                time_constants.append(tau)
            # Skip unstable eigenvalues
    
    Ts = np.array(time_constants) * 1/3600
    print("Time Constants (hours):")
    for i, t in enumerate(Ts):
        print(f"  τ{i+1}: {t:.2f}")

      
    return Ts