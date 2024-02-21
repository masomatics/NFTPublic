from scipy.special import sph_harm
import torch
from e3nn import o3
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import os


'''
The files here can be used to create Spherical Harmonics decomposition for the 
objects that are represented as 3D scattered points (Set of 3D points.)
'''


def cartesian_to_spherical(xyz):
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(xyz[:, 2] / r)    # polar angle
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])  # azimuth
    return r, theta, phi

def s2grid_to_spherical(rtp):
    f_theta_phi = rtp[0]
    theta = rtp[1]
    phi = rtp[2]    
    x = f_theta_phi * np.sin(theta) * np.cos(phi)
    y = f_theta_phi * np.sin(theta) * np.sin(phi)
    z = f_theta_phi * np.cos(theta)
    return x, y, z


def compute_real_spherical_harmonics(points, l_max, checknorm=False, mode='cartesian'):

    if mode == 'cartesian':
        r, theta, phi = cartesian_to_spherical(points)
    else:
        r, theta, phi = points[0], points[1], points[2]
    coefficients = []
    indices = []
    index = 0 
    for l in range(l_max + 1):
        lth_harmonics = [] 

        for m in range(-l, l+1, 1):
            Y = real_harmonics(m, l, theta, phi)
            if checknorm == True:
                print(m, l,'\t', sph_inprod(Y, Y, theta, phi) )
            #coeff = np.mean(r * Y[..., m].detach().numpy())
            coeff = sph_inprod(r, Y, theta, phi)
            coefficients.append(coeff)
            
            lth_harmonics.append(index) 
            index = index + 1
        indices.append(lth_harmonics)
        
    return coefficients, indices


def zero_out(indices, coefs):
    mask = np.zeros(len(coefs)) 
    mcoefs = copy.deepcopy(coefs)
    for j in indices:
        mask[j] = 1
    mcoefs = mcoefs * mask
    return mcoefs
    


def compute_real_spherical_harmonic_function(theta, phi, coefficients, l_specific=None):
    result = np.zeros(theta.shape, dtype=np.double)
    index = 0
    l_max = int(np.sqrt(len(coefficients)) - 1)
    for l in range(l_max + 1):
        if l_specific is not None and l_specific != l:
            continue

        for m in range(-l, l+1, 1):
            result += coefficients[index] * real_harmonics(m, l, theta, phi)
            #result += coefficients[index] * Y[..., m].detach().numpy()
            index += 1
    return result
 


def makeplot(coefficients, title='', l_max=25, lims = []):
    fig = plt.figure()

    #theta, phi = np.meshgrid(np.linspace(0, np.pi, 100), np.linspace(0, 2 * np.pi, 100))
    theta, phi = np.meshgrid(np.linspace(-np.pi/2, np.pi/2, 100), np.linspace(-np.pi, np.pi, 100))

    
    f_theta_phi = compute_real_spherical_harmonic_function(theta, phi, coefficients)
    x,y,z = s2grid_to_spherical([f_theta_phi, theta, phi])
    ax = fig.add_subplot(111, projection='3d')

    # Plot the spherical harmonics
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r, linewidth=0, antialiased=True, alpha=0.5)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if len(lims)>0:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])

    ax.set_title(title)
    # Show the plot
    plt.show()

def voxelfy(vals,bins=None, binsize=None):

    if type(bins)==int:
        histout = plt.hist(vals, bins=bins)
        bins = histout[1]
        binsize = (bins[1:] - bins[:-1])[0]

    rvox=  np.ceil(vals/binsize) * binsize 
    return rvox    


def numpy_isin(vec0, tensor):
    mybool = (np.sum((np.abs(vec0 - tensor)<1e-6), axis=1) ==len(vec0))*1
    return np.where(mybool==1)[0]

def remove_duplicates(fcoords): 
    fvals = fcoords[[0]]
    counts = [len(numpy_isin(fcoords[0], fcoords) )]
    for j in range(len(fcoords)):
        if len(numpy_isin(fcoords[j], fvals)) == 0:
            locs = numpy_isin(fcoords[j], fcoords) 
            fvals = np.vstack((fcoords[[j]], fvals))
            counts.append(len(locs)) 
        
    counts = np.array(counts)
    return fvals, counts
#convert the function to f(r, t, p) in the voxel form 
def voxelfy_rtp(fcoords0, bins=[], binsize=[]): 

    voxes_list = [] 
    for k in range(fcoords0.shape[1]):

        if len(binsize)  == 0 and len(bins) >0:
            vox = voxelfy(fcoords0[:,k], bins = bins[k])[:,None]

        elif len(bins) == 0 and len(binsize) >0:
            vox = voxelfy(fcoords0[:,k], binsize = binsize[k])[:,None]

        else:
            raise NotImplementedError
        
        voxes_list.append(vox)
    
    fcoords = np.hstack(voxes_list)

    fvals, counts = remove_duplicates(fcoords)


    return fvals, counts

###############################################################

'''
fxn is an array of fxn(phi, theta).
'''
def sph_inprod(fxn1, fxn2, tt, pp): 
    value = (np.mean(fxn1 * fxn2 * np.sin(tt)) * (2 * np.pi**2)  ).real
    #value = (np.mean(fxn1 * fxn2 * np.sin(tt)) ).real
    return value

'''
m <n.  Watch out for the fact that in scipy implementation, theta and phi are swapped;
i.e. theta in [0, 2pi] in their implementation. 
'''
def real_harmonics(m, n, tt, pp):
    if m > 0:
        Y=  (sph_harm(-m, n, pp, tt) + ((-1)**m) * sph_harm(m, n, pp, tt) )/np.sqrt(2) 
        Y = Y.real
    elif m < 0:
        Y=  (sph_harm(m, n, pp, tt) - ((-1)**np.abs(m)) * sph_harm(-m, n, pp, tt) )/np.sqrt(2) 
        Y = -Y.imag
    elif m == 0:
        Y = sph_harm(m, n, pp, tt).real
    
    #Y = Y *np.sqrt(2)*np.pi

    return Y.real


def vectorfy_mesh(mesh):
    mesh = mesh.reshape([mesh.shape[0]* mesh.shape[1], 1])  
    return mesh

def xyz_from_coef(coefficients, vec_form=False, gridsize=100):
    theta, phi = np.meshgrid(np.linspace(0, np.pi, gridsize), np.linspace(0, 2 * np.pi, gridsize))

    if vec_form == True:
        theta = vectorfy_mesh(theta)
        phi = vectorfy_mesh(phi)

    f_theta_phi = compute_real_spherical_harmonic_function(theta, phi, coefficients)

    x = f_theta_phi * np.sin(theta) * np.cos(phi)
    y = f_theta_phi * np.sin(theta) * np.sin(phi)
    z = f_theta_phi * np.cos(theta) 



    return np.hstack([x, y, z])



def fvals2_fr(fvals, counts):
    fvalsR = fvals[:, 0]
    rdup, r_counts = remove_duplicates(fvals[:, [0]])
    rdup = np.sort(rdup.reshape(len(rdup)))

    frs = []
    for myr in rdup:
        idx_slice = tuple(np.where(fvalsR == myr)[0])

        fr_angle = fvals[idx_slice, :]
        fr_counts = counts[np.ix_(idx_slice)][:, None]
        fr_angle = fr_angle[:, (1,2)]

        fr = np.hstack([fr_counts, fr_angle])
        frs.append(fr)

    return frs, rdup


#binsize, bins are to be specified in the order of R, Theta, and PI with 
#theta ranging from 0~pi,  phi ranging from 0~2pi.
#Returns Counts, theta, phi for EACH R
def obtain_fr(points, binsize=[], bins=[]):

    r,t,p = cartesian_to_spherical(points)
    fcoords = np.stack([r,t,p]).transpose()
    fvals, counts = voxelfy_rtp(fcoords, binsize=binsize, bins=bins)

    frs, rdup = fvals2_fr(fvals, counts)

    return frs, rdup



def convert_angle_to_indices(angles, binsize=[]):
    angle_idx = copy.deepcopy(angles)    
    angle_idx[:, 0] = (np.rint(angle_idx[:, 0]/binsize[1]))
    angle_idx[:, 1] = (np.rint(angle_idx[:, 1]/binsize[2] + np.floor(np.pi/binsize[2])))

    return angle_idx.astype(int)

def vox_angle_grid(binsize):
    thetaangle = np.linspace(0, np.pi, int(np.ceil(np.pi/binsize[1])*2))    
    thetaangle = np.ceil(thetaangle/binsize[1]) * binsize[1]
    thetaangle = np.sort(remove_duplicates(thetaangle[:, None])[0].transpose())

    phiangle = np.linspace(-np.pi, np.pi, int(np.ceil(np.pi/binsize[2])*2)) 
    phiangle = np.ceil(phiangle/binsize[2]) * binsize[2]
    phiangle = np.sort(remove_duplicates(phiangle[:, None])[0].transpose())

    thetagrid, phigrid = np.meshgrid(thetaangle, phiangle)

    return thetagrid, phigrid

#COMPUTES f(r, t, p)=number of points in rtp voxel.
def obtain_frgrids(points, binsize=[], rmargin=5):

    #obtain the list of fr(t, p) for each r  for the pair of (t,p) for which fr(t,p)>0.
    #I still have to assin 0 values to the fr(t, p)  with no counts.
    frs, rdup = obtain_fr(points, binsize=binsize)

    

    #Meshgrid of theta and phi. Not created from np.linspace to 
    #avoid the numerical error. Created from the unique list of angles
    thetagrid, phigrid = vox_angle_grid(binsize)

    #placeholder. Each slice corresponds to earh r value
    frs_all = np.zeros([len(rdup), thetagrid.shape[0], thetagrid.shape[1]]) 

    for j in range(len(rdup)):
        #angle index of the grids with nonzero "COUNTS"
        angles = frs[j][:, (1,2) ]
        angle_idx = convert_angle_to_indices(angles, binsize=binsize)
        
        #Countvalue
        frsvals = frs[j][:, 0]
        frs_all[tuple([j]), tuple(angle_idx[:,1]), tuple(angle_idx[:,0])] = frsvals

    #I need to fill 0 for all radius with no counts. 
    ridx = (np.rint(rdup/binsize[0])).astype(int)
    rmax = ridx[-1] + rmargin
    rall = (np.array(range(rmax))*binsize[0]).astype(float)
    frs_allstar =np.zeros([rmax, frs_all.shape[1], frs_all.shape[2]])
    frs_allstar[tuple(ridx), :, :]= frs_all

    return frs_allstar, rall, thetagrid, phigrid
    
def sphere_points(radius, gridsize=100): 

    theta, phi = np.meshgrid(np.linspace(0, np.pi, gridsize), np.linspace(0, 2 * np.pi, gridsize))

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta) 

    return x, y, z




#converr frtp volumetric fttp to xyz points.  Points are to appear where the frtp > 0.
def xyz_from_frtp(frtp, rdup_all, thetagrid, phigrid, threshold=0):
    xpts = []
    ypts = []
    zpts = [] 

    for idx in range(len(rdup_all)):
        rs= rdup_all[idx]
        phi_where, theta_where = np.where(frtp[idx]>threshold)
        thetas = thetagrid[0][np.ix_(theta_where)]
        phis = phigrid[:,0][np.ix_(phi_where)]

        xR,yR,zR = s2grid_to_spherical([rs, thetas, phis])
        if len(xR) > 0:
            xpts.append(xR)
            ypts.append(yR)
            zpts.append(zR)

    xpts = np.concatenate(xpts)
    ypts = np.concatenate(ypts)
    zpts = np.concatenate(zpts)
    return xpts, ypts, zpts


#obtain the estimation of the frtp with l_max. 
#when lth order is on, it will only extract the lth order.
def radial_spherical(frtp_all, thetagrid, phigrid, l_max, lth_order=-1, normalize=False): 

    frtp_hat = np.zeros(frtp_all.shape)

    for k in range(len(frtp_all)):
        coefRR, indices = compute_real_spherical_harmonics([frtp_all[k], thetagrid, phigrid],
                                                          l_max=l_max, 
                                                          checknorm=False,
                                                         mode='spherical')
        if lth_order > 0:
            coefRR = lth_mask(coefRR, lth_order, normalize=normalize) 
        
        frtp_hat[k] = compute_real_spherical_harmonic_function(thetagrid, phigrid, coefRR)
    return frtp_hat

#Onlt extract the lth spherical component of the coef.
def lth_mask(coef, lval, normalize=False):   

    allnorm = np.sqrt(np.sum(np.array(coef)**2)) 
    coefMask = np.zeros(len(coef))
    coefMask[(lval-1)**2: lval**2] = 1
    coef = np.array(copy.deepcopy(coef)) * coefMask 

    if normalize == True: 
        cnorm = np.sqrt(np.sum(np.array(coef)**2))
        coef = coef * allnorm/cnorm
    return coef


#Making movies for lth radial spherical harmonics. Put lth='all' to render movie that uses all spherical harmonics
def lth_order_animation(frtp_all, rdup_all, thetagrid, phigrid, lth, objname='', threshold=1.0,
l_max=7, root='./notebooks', boxsize=20, alpha=0.05, color="goldenrod", axis=True):

    dirname = f"""animation_{objname}"""
    dirname = os.path.join(root, dirname)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    
    if type(lth) != int:
        print('Using all Frequencies')
        lth = -1
    else:
        print(f"""Using frequency {lth}""")    
        l_max = lth + 1
        
    sph_hat_lth = radial_spherical(frtp_all, thetagrid, phigrid, l_max=l_max, lth_order = lth, normalize=True)

    xyzhat = xyz_from_frtp(sph_hat_lth, rdup_all, thetagrid, phigrid, threshold=threshold)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xyzhat[0], xyzhat[1], xyzhat[2], alpha=alpha, marker='o', s=20, c=color)
    ax.set_xlim(-boxsize, boxsize)
    ax.set_ylim(-boxsize, boxsize)
    ax.set_zlim(-boxsize, boxsize)
    if axis == False:
        ax.axis('off')

    def update(angle):
        if angle < 360:
            ax.view_init(elev=angle*0, azim=angle)
        else:
            ax.view_init(elev=(angle-360), azim=angle*0)
        return fig,

    filepath = os.path.join(dirname, f"""{lth}_order_animation{objname}.gif""") 
    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, 720, 10), interval=50)
    anim.save(filepath, fps=10)