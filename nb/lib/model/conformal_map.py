##########################
# model/conformal_map.py
##########################
# Programmer: Tim Tyree
# Date: 12.30.2022
import scipy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
import numpy as np
from ..utils.progress_bar import printProgressBar
from numba import njit
@njit
def jiH(pnt):
    """jiH castes a cartesion point, pnt, to a real quaternionic representation.
    pnt[0] is set to the diagonal members.
    """
    a=pnt[0]; b=pnt[1]; c=pnt[2]; d=pnt[3];
    h = np.array([
        [a, -b, -c, -d],
        [b,  a, -d,  c],
        [c,  d,  a, -b],
        [d, -c,  b,  a]])
    return h


def map_lam_real(lam,V,T,printing=True,**kwargs):
    """map_lam_real maps the (eigen)vector, lam, from the real representation of vector-quaternions to
    vetex representation of map, L,ome.

    Example Usage:
L,ome=map_lam_real(lam,V,T,printing=True)#,**kwargs)
    """
    #input: V,T,lam
    #output: L,ome
    nT=T.shape[0]
    nV=V.shape[0]
    #transcribe eigensolution to output mesh.
    ome=np.zeros(4*nV)
    # L  =scipy.sparse.csc_matrix(4*nV,4*nV) #<<Q: is this sparse faster?
    L  =np.zeros((4*nV,4*nV))
    # L  =sparse(4*nV,4*nV);
    num_steps=nT-1
    update_printbar_every=int(np.around(nT/20))
    step=0
    # for c1 in range(1,nT): # for c1=1:nT
    # for c1 in range(nT+1): # for c1=1:nT #<<< Index Error
    # for c1 in range(nT-1): # for c1=1:nT
    # #Q: is ^this the problem?
    # #A: it appears not...
    for c1 in range(nT): # for c1=1:nT
        for c2 in range(3): #for c2=1:3
            k0 = T[c1,(c2-1)%3]
            k1 = T[c1,(c2+0)%3]
            k2 = T[c1,(c2+1)%3]
            u1=V[k1]-V[k0]
            u2=V[k2]-V[k0]
            cta = np.dot (u1,u2) / np.linalg.norm (np.cross (u1,u2) )
            h=jiH(np.array([0.5*cta, 0, 0, 0]));
            # write to global vertex representation
            ooow = np.hstack([np.concatenate([h, -h]),
                              np.concatenate([-h, h])]) # [[h -h],-[h h]]
            ini=np.hstack([k1*4+plc,  k2*4+plc])
            # #ini=np.hstack([k2*4+plc,  k1*4+plc]) #global swap?
            L[np.ix_(ini,ini)] += ooow  # L(ini,ini) = L(ini,ini) + [[h -h],-[h h]]
            # ooow_= np.concatenate([ooow[1],ooow[2],ooow[0]],axis=2)
            # oooow = np.vstack([ooow_[1],ooow_[2],ooow_[0]])
            #L[np.ix_(ini,ini)] += oooow
            if k1>k2: #swap s.t. k3 is tmp
                k3=k1; k1=k2; k2=k3;
            lm1=jiH(lam[k1*4+plc])
            lm2=jiH(lam[k2*4+plc])
            edv=jiH(np.concatenate([np.array([0]),V[k2]-V[k1]]))
    #         ti1 = (lm1*edv)@lm1/3.   + (lm2*edv)@lm2/3.   + (lm1*edv)@lm2/6. + (lm2*edv)@lm1/6.
    #         ti1 = lm1*edv@lm1/3.   + lm2*edv@lm2/3.   + lm1*edv@lm2/6. + lm2*edv@lm1/6.
    #         ti1 = lm1*(edv@lm1)/3. + lm2*(edv@lm2)/3. + lm1*(edv@lm2)/6. + lm2*(edv@lm1)/6.
    #         ti1 = lm1@edv@lm1/3. + lm2@edv@lm2/3. + lm1@edv@lm2/6. + lm2@edv@lm1/6.
            #ti1 = (lm1@edv)@lm1/3.   + (lm2@edv)@lm2/3.   + (lm1@edv)@lm2/6. + (lm2@edv)@lm1/6.# <<< clearly wrong
            ti1 = (lm1.T@edv)@lm1/3.   + (lm2.T@edv)@lm2/3.   + (lm1.T@edv)@lm2/6. + (lm2.T@edv)@lm1/6.
            ti1 *= -1.
            #Q: which of ^these is right?
            #A: left to right, * and then @
            # til=lm1'*edv*lm1/3 + lm1'*edv*lm2/6 + lm2'*edv*lm1/6 + lm2'*edv*lm2/3;
    #         ome[k1*4+plc] = ome[k1*4+plc]-0.5*cta*ti1[0] #<<<wrong sign for rotations
            ome[k1*4+plc] = ome[k1*4+plc]-0.5*cta*ti1[:,0]  #<<< this one agrees with matlab? it looks like it gives the wrong sign to rotation
            #Q: which of ^these is right?
            # ome(k1*4+plc,1)=ome(k1*4+plc,1)-cta*til(:,1)/2;
    #         ome[k2*4+plc] = ome[k2*4+plc]+0.5*cta*ti1[0]#<<<wrong sign for rotations
            ome[k2*4+plc] = ome[k2*4+plc]+0.5*cta*ti1[:,0]  #<<< this one agrees with matlab? it looks like it gives the wrong sign to rotation
            #Q: which of ^these is right?
            # ome(k2*4+plc,1)=ome(k2*4+plc,1)+cta*til(:,1)/2;
            #Q: am I certain that i use ome[k2*4+plc] and not ome[k2*4+plc,0]??
        #print progress bar
        if printing:
            step+=1
            if step%update_printbar_every==0:
                printProgressBar(step+1, num_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)
    return L,ome

def map_real_lam_to_vertices(lam,V,T,printing=True,use_return_Lp=True,**kwargs):
    """map_real_lam_to_vertices does essentially a linear algebra solve to
    - compute the linear map of the solution vector, lam,
    - on triangular mesh with
        - vertices, V (N,3 numpy.array), and
        - faces, T (N,3, numpy.array)
    is this really 2 seconds per linear algebra solve? yes.
    UNLESS if you're out of virtual memory stored in swap ram.

    minimizes the least square of $ | Lp . x - ome | ^ 2 $

    kwargs are passed to scipy.sparse.linalg.lsqr

    Example Usage:
x,Lp,ome = map_real_lam_to_vertices(lam,V,T)#,printing=True,use_return_Lp=True,**kwargs)
    """
    L,ome=map_lam_real(lam,V,T,printing=printing)#,**kwargs)
    # print(f"{np.around(L[:10,:10],2)=}")
    # print(f"{ome[:6]=}")
    Lp=np.roll(np.roll(L,3,axis=0),3,axis=1) #bc of plc indexing...
    omep=np.roll(ome,3) #bc of plc indexing...
    #center input
    ome = omep.reshape((nV,4))
    ome = ome - np.broadcast_to(np.mean(ome,axis=0), shape=(nV,4), subok=False) #note subok=True would be needed for numpy-quaternion entries in ome.
    ome = ome.flatten()
    A = csc_matrix(Lp, dtype=float) #<<< looks fastest for demo rho
    # A = scipy.sparse.bsr_matrix(Lp, dtype=float) #<<< alternative sparse repr
    # A = scipy.sparse.csr_matrix(Lp, dtype=float) #<<< alternative sparse repr
    #solve the linear system in the least square
    b = np.array(ome, dtype=float)
    if printing:
        print(f"\nPerforming linear algebra solve...")
    x, istop, itn, normr = scipy.sparse.linalg.lsqr(A, b, **kwargs)[:4]
    if use_return_Lp:
        if printing:
            print(f"linear solve complete!\nthe number of iterations used for linear algebra solve: {istop=}")
            print(f"returning Lp,ome: {Lp.shape=}, {ome.shape=}")
            print(f"{np.min(L)=:7f},{np.max(L)=:7f}")
            print(f"{np.min(ome)=:7f},{np.max(ome)=:7f}")
        return x,Lp,ome
    else:
        return x,None,None


def comp_groundstate_eigenvector_sparse(E):
    """comp_groundstate_eigenvector_sparse
    appears faster than banded solvers...

    Example Usage:
lam,cnv = comp_groundstate_eigenvector_sparse(E)

    """
    # Eab = decomp_banded_matrix(E)
    E_sparse=scipy.sparse.csc_matrix(E,dtype=float)
    lam=np.zeros(4*nV)
    lam[::4]=1.
    b=lam
    # num_steps=11
    # for step in range(num_steps):
    #     cnv=lam
    #Q: what's the fastest way to compute x vs. x2 or x3
    x2 =scipy.sparse.linalg.lsqr(E_sparse,b)[0] #lam=lam/E
    # x3 =solveh_banded(Eab,b)#,lower=False)#,**kwargs) #lam =mldivide(lam,ab)
    #Q: what's the fastest way to compute x vs. x2 or x3
    # A: x2
    lam=x2.copy()
    # lam=x3.copy()
    cnv=b.copy()
    return lam,cnv

def comp_groundstate_eigenvector_inverse(E,num_steps=11,printing=True):
    """computes lam = lam . Einv num_steps times.
    comp_groundstate_eigenvector_inverse has a relatively slow run time...

    Example Usage:
lam,cnv,Einv = comp_groundstate_eigenvector_inverse(E)
res=(lam@E)/lam;
print(f'mean: {np.mean(res):e}, var:  {np.var(res):e}, delta: {np.linalg.norm(cnv-lam):e}')
    """
    #sparse soln:
    # mean: -1.825385e-04, var:  1.688050e-04, delta: 2.156203e+05
    # Q: how does lam look?
    # A: great!
    #banded soln: ?? second run time.... does it crash the kernel?
    # mean: ??
    # Q: how does lam look?
    # A: ??
    #omverse soln:
    #3 minute runtime for 11 epochs
    #isn't this supposed to be faster
    #input: E,nV=E.shape[0]/4
    #output: lam
    lam=np.zeros(4*nV)
    lam[::4]=1.
    # lam[1::4]=1.
    # lam[-1::4]=1.
    # lam[3::4]=1.
    # lam/=np.linalg.norm(lam)
    # lam+=1.
    #Q: is ^this right?
    #Q is this better?
    # ab = decomp_banded_matrix(E)
    # ab = decomp_banded_matrix(E)
    # E_sparse=scipy.sparse.csc_matrix(E,dtype=float)
    Einv=np.linalg.inv(E)
    for step in range(num_steps):
        cnv=lam

        lam = lam@Einv
        #lam =scipy.sparse.linalg.lsqr(E_sparse,lam)[0] #lam=lam/E
        #lam =solveh_banded(ab,lam)#,lower=False)#,**kwargs) #lam =mldivide(lam,ab)
        lam/=np.linalg.norm(lam);
        #print progress bar
        if printing:
            step+=1
            if step%update_printbar_every==0:
                printProgressBar(step+1, num_steps, prefix = 'Progress:', suffix = 'Complete', length = 50)
    return lam,cnv,Einv


#####################################
# helper functions
#####################################
def comp_trimesh_curvature(mesh,radius=1./(2.*np.pi)):
    """
    Q: is this the general identity map by default?
    A: yes! yes, it is.

    Example Usage:
rho = comp_trimesh_curvature(mesh,radius=1./(2.*np.pi))
    """
    tria=mesh.faces
    mean_curvature_values = trimesh.curvature.\
                            discrete_mean_curvature_measure(mesh,
                                                            points=mesh.vertices,
                                                            radius=radius)
    # rho_vertex_values = mean_curvature_values/radius**2
    rho_vertex_values = mean_curvature_values/radius**2/2
    rho_vertex_values[tria].shape
    #is this the identity map?
    rho = np.mean(rho_vertex_values[tria],axis=1)
    return rho
