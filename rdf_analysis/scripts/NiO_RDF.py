import numpy as np
from numba import jit
import time
import MDAnalysis as mda

#define the parameters
nbins = 200
dumpfile='wrapped.lammpstrj'
dump_freq=1000
dt=0.001    # ps 
nblocks = 5 # number of blocks
bin_start = 0.0
bin_end = 10.0 # Angstrom

@jit(nopython=True)
def calculate_cross_term_ni_o(coords_Ni, coords_O, lbox, nbins, bin_start, bin_end):
    n1 = coords_Ni.shape[0]  # number of Ni atoms
    n2 = coords_O.shape[0] # number of O atoms
    npairs = n1 * n2
    rho = npairs / (lbox[0] * lbox[1] * lbox[2])

    dists = []
    for i in range(n1):
        # Calculate x/y/z component distances
        x_dist = np.abs(coords_Ni[i, 0] - coords_O[:, 0])
        y_dist = np.abs(coords_Ni[i, 1] - coords_O[:, 1])
        z_dist = np.abs(coords_Ni[i, 2] - coords_O[:, 2])
        # PBC correction
        x_dist = np.where(x_dist > lbox[0] / 2.0, x_dist - lbox[0], x_dist)
        y_dist = np.where(y_dist > lbox[1] / 2.0, y_dist - lbox[1], y_dist)
        z_dist = np.where(z_dist > lbox[2] / 2.0, z_dist - lbox[2], z_dist)
        # Calculate distances
        dist = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
        dists.extend(dist)
    # Histogram all atom distances
    dists = np.array(dists)
    dists = dists[dists < bin_end] # all distance < L/2

    hist, edges = np.histogram(dists, nbins, (bin_start, bin_end))
    r = edges[:-1]
    dr = np.diff(edges)
    r_sq = r**2
    norm = 4.0 * np.pi * rho * r_sq * dr # normalization factor
    g_r = hist / norm # normalize histogram to give g(r)
    return r, g_r

def coordination_number(r, g_r, rho, r_min):
    """
    CN = 4πrho * ∫_0^{r_min} g(r) r^2 dr

    rho : number density of O atoms (N_O / V), Å^-3
    r_min : first minimum after main Ni-O peak (Å)
    """
    mask = (r > 0.0) & (r <= r_min) # Mask r from 0 → r_min
    r_selected = r[mask] # using this r_selected to calculate the integral
    g_selected = g_r[mask] # using this g_selected to calculate the integral
    integrand = 4.0 * np.pi * rho * g_selected * r_selected**2
    CN = np.trapezoid(integrand, r_selected)
    return CN

print('---------- SYSTEM/TRAJ INFO ----------')
u = mda.Universe(dumpfile, format='LAMMPSDUMP', lengthunit='A', timeunit='ps', dt=(dump_freq*dt))
natoms = u.atoms.n_atoms
total_nframes = len(u.trajectory)

# Set the frame range for RDF calculation
frame_start = 1000
frame_end = 6000
sampled_nframes = frame_end - frame_start + 1
nframesperblock = int(np.floor(sampled_nframes / nblocks))

print('natoms', natoms)
print('total nframes', total_nframes, '(', total_nframes*dump_freq*dt, 'ps)')
print('sampled nframes:', sampled_nframes, '(', sampled_nframes*dump_freq*dt, 'ps)')
print('nblocks', nblocks)
print('nframes per block', nframesperblock)
print('r range:', bin_start, '-', bin_end)

print('-------------- CALC RDF --------------')
start = time.time()
gr_ni_o_total = np.zeros((sampled_nframes, nbins))

frame_idx = 0
Ni_atoms = u.select_atoms('type 3')
O_atoms = u.select_atoms('type 1')
for i, ts in enumerate(u.trajectory):
    # Only process frames in the specified range
    if i < frame_start:
        continue
    if i > frame_end:
        break

    lbox = ts.dimensions[:3] 
        
    pos_Ni = Ni_atoms.positions
    pos_o = O_atoms.positions

    r, gr_ni_o_total[frame_idx, :] = calculate_cross_term_ni_o(pos_Ni, pos_o, lbox, nbins, bin_start, bin_end)

    frame_idx += 1

end = time.time()
print('time:', end - start)

# Simple averaging with blocks
gr_ni_o_block_avg = np.zeros((nblocks, nbins))

for block in range(nblocks):
    start_idx = block * nframesperblock
    end_idx = (block + 1) * nframesperblock
    gr_ni_o_block_avg[block, :] = np.mean(gr_ni_o_total[start_idx:end_idx, :], axis=0)

gr_ni_o_mean = np.mean(gr_ni_o_block_avg, axis=0)
gr_ni_o_std = np.std(gr_ni_o_block_avg, axis=0)
gr_ni_o_ste = gr_ni_o_std / np.sqrt(nblocks)  # Standard error

r_block = r  # This is the correct r to use for averaged RDFs

header = 'r(Angstrom) g(r) g(r)_error'
np.savetxt('Ni-O.txt', 
           np.column_stack((r_block, gr_ni_o_mean, gr_ni_o_ste)), 
           header=header, 
           fmt='%.6f')

print(f"RDF saved to Ni-O.txt")

# ===================== COORDINATION NUMBER (Li–O in EC) ===================== #
rho_NiO = O_atoms.n_atoms / (lbox[0] * lbox[1] * lbox[2])
print('rho_NiO =', rho_NiO)

r_min = 2.8

CN_blocks = np.zeros(nblocks)
for block in range(nblocks):
    CN_blocks[block] = coordination_number(r_block, gr_ni_o_block_avg[block, :], rho_NiO, r_min)

CN_mean = np.mean(CN_blocks)
CN_std  = np.std(CN_blocks, ddof=1)      # sample std
CN_ste  = CN_std / np.sqrt(nblocks)
print("CN_NiO =", CN_mean, "+/-", CN_ste)







   
