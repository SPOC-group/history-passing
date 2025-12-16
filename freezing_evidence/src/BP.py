import numpy as np
import itertools
from utils.rules import *
from src.population_dynamics_torch import *

import json
import csv
import ast
import os
from scipy import special
import sympy as sym

class BP:
    def __init__(self, rule, mu=0, tol=1e-12, max_iter=10000, max_iter_SP=500, tol_SP=1e-8, damping_parameter=0.8, init_message=None, SP_init='hard', seed=None, device='cuda'):
        self.rule=rule
        self.mu=mu
        self.tol=tol
        self.max_iter=max_iter
        self.max_iter_SP=max_iter_SP
        self.tol_SP=tol_SP
        self.damping_parameter=damping_parameter
        self.rng=np.random.default_rng()
        if seed is not None:
            self.rng=np.random.default_rng(seed)
        
        self.psi=init_message
        if self.psi is None:
            self.psi=self.rng.uniform(size=(2,2))
            self.psi=self.psi/np.sum(self.psi)
        self.device=device
            
        self.psi_old=self.psi.copy()
        
        self.d=len(rule)-1
        self.permutations=np.array(list(itertools.product([0,1], repeat=self.d-1)))

        # all possible warnings:
        self.all_warnings=np.array(list(itertools.product([0,1], repeat=4))).reshape((16,2,2))
        self.all_warnings=(self.all_warnings==1)

        # SP survey initialization:
        self.eta=np.zeros(16)
        if SP_init=='hard':
            self.eta[[1,2,4,8]]=1.
        elif SP_init=='noisy hard':
            self.eta[[1,2,4,8]]=1.
            self.eta[15]=1e-4
        else:
            self.eta=self.rng.uniform(size=16)

        self.eta=self.eta/np.sum(self.eta)
        self.eta_old=self.eta.copy()

        # All possible d warnings permutation for d-1 neighbors:
        self.warning_permutations=np.array(list(itertools.product(np.arange(16), repeat=self.d-1)))



        # An array containing the number of permutations for the d-1 neighbors
        self.num_perm_BP=np.zeros(self.d)
        for k in range(self.d):
            self.num_perm_BP[k]=special.comb(N=self.d-1, k=k, exact=True, repetition=False)

        # An array containing the number of permutations for the d neighbors
        self.num_perm_obs=np.zeros(self.d+1)
        for k in range(self.d+1):
            self.num_perm_obs[k]=special.comb(N=self.d, k=k, exact=True, repetition=False)


        
        self.phi=None
        self.rho=None
        self.s=None
        
        self.stability=None
        self.physical=None

        self.stability_AT=None
        self.linear_stability=None
        self.non_linear_stability=None

        
        self.fixed_points=None
        self.all_phi=None
        self.all_rho=None
        self.all_s=None
        
        self.all_stabilities=None
        self.all_physical=None

        self.phi_SP=None
        self.stability_SP=None
        self.phi_SP_noisy=None
        
             
        
    def __repr__(self):
        description="Instance of class \'BP\'\nRule : "+str(self.rule)+"\nμ =  "+str(self.mu)
        if self.phi is not None:
            description+='\nφ = '+str(self.phi)
            description+='\nρ = '+str(self.rho)
            description+='\ns = '+str(self.s)
            if self.stability_AT is not None:
                description+='\nStability noisy fixed point in population dynamics: '+str(self.stability_AT)
            if self.linear_stability is not None:
                description+='\nLinear susceptibility stability: '+str(self.linear_stability)
                description+='\nNon-linear susceptibility stability: '+str(self.non_linear_stability)
        if self.phi_SP is not None:
            description+='\nφ SP: '+str(self.phi_SP)
        if self.stability_SP is not None:
            description+='\nStability SP: '+str(self.stability_SP)
        if self.phi_SP_noisy is not None:
             description+='\nφ SP noisy: '+str(self.phi_SP_noisy)
          

        return description
    
    #=============================================================================================================================
    # Belief propagation and observables
    #=============================================================================================================================         
    

    def respect_rule(self, i, j, rest_config):
        outer_density=j+np.sum(rest_config)
        if self.rule[outer_density]=='0':
            return True if i==0 else False
        elif self.rule[outer_density]=='1':
            return True if i==1 else False
        elif self.rule[outer_density]=='+':
            return True
        elif self.rule[outer_density]=='-':
            return False
    
    def respect_rule_v2(self, i, outer_density):
        if self.rule[outer_density]=='0':
            return True if i==0 else False
        elif self.rule[outer_density]=='1':
            return True if i==1 else False
        elif self.rule[outer_density]=='+':
            return True
        elif self.rule[outer_density]=='-':
            return False
                        
    def diff(self):
        return np.linalg.norm(self.psi-self.psi_old)

    def diff_SP(self):
        return np.linalg.norm(self.eta-self.eta_old)
    
    def step(self):
        self.psi_old=self.psi.copy()
        psi_new=np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                for num_alive_k in range(self.d):
                    psi_new[i,j]+=self.num_perm_BP[num_alive_k]*np.exp(self.mu*i)*self.respect_rule_v2(i, j+num_alive_k)*self.psi[0,i]**(self.d-1-num_alive_k)*self.psi[1,i]**(num_alive_k)
        Z=np.sum(psi_new)
        if Z!=0:
            psi_new=psi_new/np.sum(psi_new)
        else:
            psi_new=np.array([[0.25, 0.25],[0.25, 0.25]])

        
        self.psi=self.damping_parameter*self.psi+(1-self.damping_parameter)*psi_new

    def step_symmetric(self):
        self.psi_old=self.psi.copy()
        psi_new=np.zeros((2,2))


        # not implemented in the more efficient way here as in step
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    mult=1
                    for k in perm:
                        if self.rng.random()<0.5:
                            mult*=np.flip(self.psi,axis=(0,1))[k,i]
                        else:
                            mult*=self.psi[k,i]
                    psi_new[i,j]+=np.exp(self.mu*i)*self.respect_rule(i, j, perm)*mult
        if np.sum(psi_new)!=0:
            psi_new=psi_new/np.sum(psi_new)
        else:
            psi_new=np.array([[0.25, 0.25],[0.25, 0.25]])
        
        self.psi=self.damping_parameter*self.psi+(1-self.damping_parameter)*psi_new

    # Warning propagation:
    # Note: in this case (mu s_i) does nothing
    def step_WP(self):
        self.psi_old=self.psi.copy()
        psi_new=np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                for num_alive_k in range(self.d):
                    psi_new[i,j]+=self.num_perm_BP[num_alive_k]*np.exp(self.mu*i)*self.respect_rule_v2(i, j+num_alive_k)*self.psi[0,i]**(self.d-1-num_alive_k)*self.psi[1,i]**(num_alive_k)

        psi_new=(psi_new!=0.)
        if np.sum(psi_new)!=0:
            psi_new=psi_new/np.sum(psi_new)
        else:
            psi_new=np.array([[0.25, 0.25],[0.25, 0.25]])
        
        self.psi=self.damping_parameter*self.psi+(1-self.damping_parameter)*psi_new
        


        
    def warning_evolution(self, warning, warnings):
        new_warning=np.zeros((2,2), dtype=bool)
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    if self.respect_rule_v2(i,j+np.sum(perm)):
                        respect_warning=True
                        for k in range(self.d-1):
                            if not warnings[k,perm[k],i]:
                                respect_warning=False
                                break
                        if respect_warning:
                            new_warning[i,j]=True
                if new_warning[i,j]!=warning[i,j]:
                    return False
        return True

    def warning_consistency(self, warning, warnings):
        for i in range(2):
            for j in range(2):
                for perm in self.permutations:
                    if self.respect_rule_v2(i,j+np.sum(perm)):
                        respect_warning=warning[j,i]
                        if respect_warning:
                            for k in range(self.d-1):
                                if not warnings[k,perm[k],i]:
                                    respect_warning=False
                                    break
                        if respect_warning:
                            return True
        return False

    def step_SP(self):
        self.eta_old=self.eta.copy()
        new_eta=np.zeros(16)
        for w in range(1,16):
            for warnings in self.warning_permutations:
                mult=1
                for w_k in warnings:
                    if self.eta[w_k]==0:
                        mult=0
                        break
                    mult*=self.eta[w_k]
                if mult!=0:
                    new_eta[w]+=self.warning_evolution(self.all_warnings[w], self.all_warnings[warnings])*mult
        new_eta[0]=0.
        Z=np.sum(new_eta)
        if Z!=0:
            new_eta=new_eta/Z
        else:
            # print('Warning: all the new SP messages were 0. Setting them to uniform distribution ...')
            new_eta=np.ones(16)/16.
        

        self.eta=self.damping_parameter*self.eta+(1-self.damping_parameter)*new_eta

    
    def update_observables(self):
        phi_=0
        phi__=0
        numerator=0
        for i in range(2):
            for num_alive_k in range(self.d+1):
                phi_+=self.num_perm_obs[num_alive_k]*np.exp(self.mu*i)*self.respect_rule_v2(i, num_alive_k)*self.psi[0,i]**(self.d-num_alive_k)*self.psi[1,i]**(num_alive_k)
                numerator+=i*self.num_perm_obs[num_alive_k]*np.exp(self.mu*i)*self.respect_rule_v2(i, num_alive_k)*self.psi[0,i]**(self.d-num_alive_k)*self.psi[1,i]**(num_alive_k)
            for j in range(2):
                phi__+=self.psi[i,j]*self.psi[j,i]

        self.phi=np.log(phi_)-self.d/2*np.log(phi__)
        

        if numerator==0:
            self.rho=0
        else:
            self.rho=numerator/phi_
            
        self.s=self.phi-self.mu*self.rho


    def update_observables_SP(self):
        Z_i=0
        Z_ij=0
        for w in range(16):
            # Can improve this, the warnings are symmetric
            for warnings in self.warning_permutations:
                mult=self.eta[w]
                if mult!=0:
                    for w_k in warnings:
                        if self.eta[w_k]==0:
                            mult=0
                            break
                        mult*=self.eta[w_k]
                    if mult!=0:
                        Z_i+=self.warning_consistency(self.all_warnings[w], self.all_warnings[warnings])*mult
            for w_j in range(16):
                if np.any(np.logical_and(self.all_warnings[w]==True, self.all_warnings[w_j].T==True)):
                    Z_ij+=self.eta[w]*self.eta[w_j]
                    
        self.phi_SP=np.log(Z_i)-self.d/2*np.log(Z_ij)
        return self.phi_SP
        
    def marginals(self):
        marginals=np.zeros((2,2))
        marginals[0,0]=self.psi[0,0]**2
        marginals[1,0]=self.psi[1,0]*self.psi[0,1]
        marginals[0,1]=self.psi[0,1]*self.psi[1,0]
        marginals[1,1]=self.psi[1,1]**2
        return marginals/np.sum(marginals) if np.sum(marginals)!=0 else np.array([[0.25, 0.25],[0.25, 0.25]])
    
    def run(self,max_iter=None, tol=None, verbose=False):
        if max_iter is None:
            max_iter=self.max_iter
        if tol is None:
            tol=self.tol
        for i in range(max_iter):
            self.step()
            if self.diff()<tol:
                break
        if i==max_iter-1:
            print('No convergence reached for rule ', self.rule, 'after ', i,' steps. Final error: ', self.diff())
        else:
            if verbose:
                print('Converged after ', i, ' steps. Final error: ', self.diff())
        self.update_observables()


    def run_WP(self,max_iter=None, tol=None, verbose=False):
        if max_iter is None:
            max_iter=self.max_iter
        if tol is None:
            tol=self.tol
        for i in range(max_iter):
            self.step_WP()
            if self.diff()<tol:
                break
        if i==max_iter-1:
            print('No convergence reached for rule ', self.rule, 'after ', i,' steps. Final error: ', self.diff())
        else:
            if verbose:
                print('Converged after ', i, ' steps. Final error: ', self.diff())
        self.update_observables()

    def run_SP(self,max_iter=None, tol=None, verbose=False):
        if max_iter is None:
            max_iter=self.max_iter_SP
        if tol is None:
            tol=self.tol_SP
        for i in range(max_iter):
            self.step_SP()
            diff_SP=self.diff_SP()
            if verbose>=2 and i%5==0:
                print('Iteration : ', i,' Diff : ', diff_SP)
            if diff_SP<tol:
                break
        if i==max_iter-1:
            print('No convergence reached for SP for rule ', self.rule, 'after ', i,' steps. Final error: ', self.diff_SP())
        else:
            if verbose:
                print('Converged after ', i, ' steps. Final error: ', self.diff_SP())
        self.update_observables_SP()
        
    def run_symmetric(self,max_iter=None, tol=None, verbose=False):
        if max_iter is None:
            max_iter=self.max_iter
        if tol is None:
            tol=self.tol
        for i in range(max_iter):
            self.step_symmetric()
            if self.diff()<tol:
                break
        if i==max_iter-1:
            print('No convergence reached for rule ', self.rule, 'after ', i,' steps. Final error: ', self.diff())
        else:
            if verbose:
                print('Converged after ', i, ' steps. Final error: ', self.diff())
        self.update_observables()
        

    def is_stable(self,fixed_point=None, precision=8, noise=1e-9, random_noise=True, num_tests=10 ):
        if fixed_point is None:
            fixed_point=self.psi
        old_psi=self.psi
        
        if random_noise:
            for i in range(num_tests):
                noisy_fixed_point=fixed_point+np.abs(self.rng.normal(0,noise,(2,2)))
                noisy_fixed_point/=np.sum(noisy_fixed_point)
                self.psi=noisy_fixed_point
                self.run()
                if np.array_equal(np.round(self.psi,precision), np.round(fixed_point,precision)):
                    self.stability=True
                    self.psi=old_psi
                    self.update_observables()
                    if i>=1:
                        print("Warning: ", i+1, " stability checks were done before finding a stable noisy fixed point for rule ", self.rule, " .")
                    return True

            self.stability=False
            self.psi=old_psi
            self.update_observables()
            return False
        else:
            noisy_fixed_point=fixed_point+noise*np.ones((2,2))
            noisy_fixed_point/=np.sum(noisy_fixed_point)
            self.psi=noisy_fixed_point
            self.run()
            if np.array_equal(np.round(self.psi,precision), np.round(fixed_point,precision)):
                self.stability=True
                self.psi=old_psi
                self.update_observables()
                return True
            else:
                self.stability=False
                self.psi=old_psi
                self.update_observables()
                return False

    def is_stable_AT(self, fixed_point=None, noise=1e-6, M=10000000):
        if fixed_point is None:
            fixed_point=self.psi
        planted=np.tile(fixed_point, (10000000,1,1))
        pop=population_dynamics_torch(rule=self.rule,M=M, tol=15.,planted_messages=planted, noise_init_messages=noise)
        pop.run()
        if np.linalg.norm(fixed_point-np.mean(pop.population.cpu().numpy(), axis=0, dtype=np.float64))<1e-6: # Float 64 needed since we take the mean over many points
            self.stability_AT=True
            return True
        else:
            self.stability_AT=False
            return False

    def analytic_stability(self, fixed_point=None,verbose=0): 
        psi_RS=None
        if fixed_point is None:
            psi_RS=self.psi
        else:
            psi_RS=fixed_point
        psi_00=sym.Symbol('psi_00')
        psi_01=sym.Symbol('psi_01')
        psi_10=sym.Symbol('psi_10')
        psi_11=sym.Symbol('psi_11')
        psi=np.array([[psi_00, psi_01], [psi_10,psi_11]])
        psi_new=[[0,0],[0,0]]
        Z=0
        for i in range(2): 
            for j in range(2):
                for perm in self.permutations:
                    mult=1
                    if self.respect_rule(i,j,perm):
                        for k in perm:
                            mult*=psi[k,i]
                        Z+=np.exp(self.mu*i)*mult
                        psi_new[i][j]+=np.exp(self.mu*i)*mult
        if Z!=0:
            for i in range(2):
                for j in range(2):
                    psi_new[i][j]/=Z
        else:
            print('Warning: normalization for the linear stability is 0')
            
        T=[[[] for _ in range(4)] for _ in range(4)]
        T_eval=np.zeros((4,4))
        #configs=[[1,1],[1,0],[0,1],[0,0]]
        configs=[[0,0],[0,1],[1,0],[1,1]]
        for idx_1, conf_1 in enumerate(configs):
            for idx_2, conf_2 in enumerate(configs):
                T[idx_1][idx_2]=sym.diff(psi_new[conf_1[0]][conf_1[1]], psi[conf_2[0], conf_2[1]])
                T_eval[idx_1, idx_2]=float(T[idx_1][idx_2].subs([(psi_00,psi_RS[0,0]),(psi_01,psi_RS[0,1]),(psi_10,psi_RS[1,0]),(psi_11,psi_RS[1,1])]))
        
        for i in range(4):
            for j in range(4):
                if configs[i][0]!=configs[j][1]:
                    pass
                    #T_eval[i,j]=0.
        
        eigs=np.abs(np.linalg.eigvals(T_eval))
        lambda_max=np.max(eigs)
        if verbose>0:
            print('max eigenvalue: '+str(lambda_max))
        self.linear_stability= True if lambda_max<=1 else False
        self.non_linear_stability=True if 1/(self.d-1)*lambda_max**2 <=1 else False
        return self.linear_stability, self.non_linear_stability, T, T_eval, lambda_max, psi_new

    def is_stable_SP(self, noise=1e-4):
        old_eta=self.eta.copy()
        old_phi_SP=self.phi_SP
        self.eta[[1,2,4,8]]=1.
        self.eta[15]=noise
        self.eta=self.eta/np.sum(self.eta)
        self.eta_old=self.eta
        self.run_SP()
        if np.linalg.norm(old_eta-self.eta)<=1e-4:
            self.stability_SP=True
        else:
            self.stability_SP=False
            self.update_observables_SP()
            self.phi_SP_noisy=self.phi_SP
        self.eta=old_eta.copy()
        self.eta_old=old_eta.copy()
        self.phi_SP=old_phi_SP
        self.update_observables_SP()
        return self.stability_SP
            
            
    
    def is_physical(self, fixed_point=None):
        
        if fixed_point is None:
            fixed_point=self.psi
        if (fixed_point[0,0]+fixed_point[1,0]>=1-1e-6 and fixed_point[1,0]>1e-6) or (fixed_point[0,1]+fixed_point[1,1]>=1-1e-6 and fixed_point[0,1]>1e-6): # np.array_equal(np.round(fixed_point,3), np.array([[0.,1],[0,0]])) or np.array_equal(np.round(fixed_point,3), np.array([[0.,0],[1,0]])):
            self.physical=False
            return False
        elif self.s>(np.log(2)+1e-4):
            print("Warning, entropy > log(2) even if fixed point physical !")
            self.physical=False
            return False
        else:
            self.physical=True
            return True
        
    #=============================================================================================================================
    # Grid search for all fixed points
    #============================================================================================================================= 
            
    def find_all_fixed_points_not_vectorized(self, grid_discretisation=15, precision=8, num_random_search=0):
        self.fixed_points=[]
        self.all_phi=[]
        self.all_rho=[]
        self.all_s=[]
        self.all_stabilities=[]
        self.all_physical=[]
        for i in np.linspace(0,1, grid_discretisation):
            for j in np.linspace(0,1-i, grid_discretisation):
                for k in np.linspace(0,1-i-j, grid_discretisation):
                    self.psi=np.array([[i,j],[k,1-i-j-k]])
                    self.run()
                    if not self.fixed_points or not np.any(np.all(np.round(self.psi, precision) == np.round(self.fixed_points, precision), axis=(1,2))):
                        self.fixed_points.append(self.psi)
                        self.update_observables()
                        self.all_phi.append(self.phi)
                        self.all_rho.append(self.rho)
                        self.all_s.append(self.s)
        for _ in range(num_random_search):
            self.psi=self.rng.uniform(size=(2,2))
            self.run()
            if not self.fixed_points or not np.any(np.all(np.round(self.psi, precision) == np.round(self.fixed_points, precision), axis=(1,2))):
                self.fixed_points.append(self.psi)
                self.update_observables()
                self.all_phi.append(self.phi)
                self.all_rho.append(self.rho)
                self.all_s.append(self.s)
                self.all_stabilities.append(self.is_stable())
                self.all_physical.append(self.is_physical())
                
        if any(self.all_physical):
            self.phi=np.max(np.array(self.all_phi)[self.all_physical])
            idx_max_stable=self.all_phi.index(self.phi)
            self.psi=self.fixed_point[idx_max_stable]
            self.rho=self.all_rho[idx_max_stable]
            self.s=self.all_s[idx_max_stable]
            self.psi=self.fixed_points[idx_max_stable]
            self.stability=self.all_stabilities[idx_max_stable]
            self.physical=True
        else:
            # print("No physical fixed point found !")
            self.phi=max(self.all_phi)
            idx_max=self.all_phi.index(self.phi)
            self.psi=self.fixed_points[idx_max]
            self.rho=self.all_rho[idx_max]
            self.s=self.all_s[idx_max]
            self.psi=self.fixed_points[idx_max]
            self.stability=self.all_stabilities[idx_max]
            self.physical=False

                
                
    def find_all_fixed_points(self, grid_discretisation=20, precision=6, num_random_search=0, verbose=0):
        messages=[]
        for i in np.linspace(0,1,grid_discretisation):
            for j in np.linspace(0,1-i,grid_discretisation):
                for k in np.linspace(0,1-i-j,grid_discretisation):
                    messages.append(np.array([[i,j],[k, 1-i-j-k]]))
        for _ in range(num_random_search):
            messages.append(self.rng.uniform(size=(2,2)))
        messages=np.array(messages)
        num_samples=grid_discretisation**3+num_random_search
        old_messages=None
        convergence=False

                
        for it in range(self.max_iter):
            old_messages=messages.copy()
            new_messages=np.zeros((num_samples,2,2))
            for i in range(2):
                for j in range(2):
                    for num_alive_k in range(self.d):
                        new_messages[:,i,j]+=self.num_perm_BP[num_alive_k]*np.exp(self.mu*i)*self.respect_rule_v2(i, j+num_alive_k)*messages[:,0,i]**(self.d-1-num_alive_k)*messages[:,1,i]**(num_alive_k)
                   
            Z=np.sum(new_messages, axis=(1,2))
            new_messages[np.where(Z==0)]=np.array([[0.25,0.25],[0.25,0.25]])
            Z=np.where(Z!=0,Z,1)
            new_messages/=Z.repeat(4).reshape(num_samples,2,2)
            messages=self.damping_parameter*messages+(1-self.damping_parameter)*new_messages
            if np.max(np.linalg.norm(messages-old_messages, axis=(1,2)))<self.tol:
                if verbose>=1:
                    print("Convergence reached for each starting initialisation after ", it, " steps !")
                convergence=True
                break

        if not convergence:
            print("Not every starting initialisation has converged after ", self.max_iter, " steps.")
                
        if verbose>=1:
            print('Max error: ', np.max(np.linalg.norm(messages-old_messages, axis=(1,2))))
            
        _, idx_unique=np.unique(np.round(messages, precision), axis=0, return_index=True)
        self.fixed_points=messages[idx_unique]
        self.all_phi=[]
        self.all_rho=[]
        self.all_s=[]
        self.all_stabilities=[]
        self.all_physical=[]
        for fixed_point in self.fixed_points:
            self.psi=fixed_point
            self.update_observables()
            self.all_phi.append(self.phi)
            self.all_rho.append(self.rho)
            self.all_s.append(self.s)
            self.all_stabilities.append(self.is_stable())
            self.all_physical.append(self.is_physical())

            
        if any(self.all_physical):
            self.phi=np.max(np.array(self.all_phi)[self.all_physical])
            idx_max_stable=self.all_phi.index(self.phi)
            self.psi=self.fixed_points[idx_max_stable]
            self.rho=self.all_rho[idx_max_stable]
            self.s=self.all_s[idx_max_stable]
            self.psi=self.fixed_points[idx_max_stable]
            self.stability=self.all_stabilities[idx_max_stable]
            self.physical=True
        else:
            # print("No physical fixed point found !")
            self.phi=max(self.all_phi)
            idx_max=self.all_phi.index(self.phi)
            self.rho=self.all_rho[idx_max]
            self.s=self.all_s[idx_max]
            self.psi=self.fixed_points[idx_max]
            self.stability=self.all_stabilities[idx_max]
            self.physical=False
            self.psi=fixed_point[idx_max]

        
    
    # More efficient for a grid bigger than ~40
    def find_all_fixed_points_torch(self, grid_discretisation=100, precision=6, num_random_search=0, verbose=0):
        import torch
        if self.device=='cuda':
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
            
        messages=[]
        for i in np.linspace(0,1,grid_discretisation):
            for j in np.linspace(0,1-i,grid_discretisation):
                for k in np.linspace(0,1-i-j,grid_discretisation):
                    messages.append(np.array([[i,j],[k, 1-i-j-k]]))
        for _ in range(num_random_search):
            messages.append(self.rng.uniform(size=(2,2)))
        messages=torch.from_numpy(np.array(messages)).to(self.device)
        num_samples=grid_discretisation**3+num_random_search
        old_messages=None
        convergence=False
        
        for it in range(self.max_iter):
            old_messages=messages.clone()
            new_messages=torch.zeros((num_samples,2,2))
            for i in range(2):
                for j in range(2):
                    for num_alive_k in range(self.d):
                        new_messages[:,i,j]+=self.num_perm_BP[num_alive_k]*np.exp(self.mu*i)*self.respect_rule_v2(i, j+num_alive_k)*messages[:,0,i]**(self.d-1-num_alive_k)*messages[:,1,i]**(num_alive_k)
                        
            Z=torch.sum(new_messages, axis=(1,2))
            new_messages[torch.where(Z==torch.tensor([0.]))]=torch.tensor([[0.25,0.25],[0.25,0.25]])
            Z=torch.where(Z!=torch.tensor([0.]),Z,torch.tensor([1.]))
            new_messages/=Z.repeat_interleave(4).reshape(num_samples,2,2)
            messages=self.damping_parameter*messages+(1-self.damping_parameter)*new_messages
            if torch.max(torch.linalg.matrix_norm(messages-old_messages, dim=(1,2)))<self.tol:
                if verbose>=1:
                    print("Convergence reached for each starting initialisation after ", it, " steps !")
                convergence=True
                break


        if not convergence:
            print("Not every starting initialisation has converged after ", self.max_iter, " steps.")
                
        if verbose>=1:
            print('Max error: ', (torch.max(torch.linalg.norm(messages-old_messages, dim=(1,2)))).item())
        
        messages_np=messages.cpu().detach().numpy()
        _, idx_unique=np.unique(np.round(messages_np, precision), axis=0, return_index=True)
        self.fixed_points=messages_np[idx_unique]
        self.all_phi=[]
        self.all_rho=[]
        self.all_s=[]
        self.all_stabilities=[]
        self.all_physical=[]

        for fixed_point in self.fixed_points:
            self.psi=fixed_point
            self.update_observables()
            self.all_phi.append(self.phi)
            self.all_rho.append(self.rho)
            self.all_s.append(self.s)
            self.all_stabilities.append(self.is_stable())
            self.all_physical.append(self.is_physical())

        if any(self.all_physical):
            self.phi=np.max(np.array(self.all_phi)[self.all_physical])
            idx_max_stable=self.all_phi.index(self.phi)
            self.psi=self.fixed_points[idx_max_stable]
            self.rho=self.all_rho[idx_max_stable]
            self.s=self.all_s[idx_max_stable]
            self.psi=self.fixed_points[idx_max_stable]
            self.stability=self.all_stabilities[idx_max_stable]
            self.physical=True
        else:
            print("No physical fixed point found for rule "+str(self.rule)+" !")
            self.phi=max(self.all_phi)
            idx_max=self.all_phi.index(self.phi)
            self.psi=fixed_point[idx_max]
            self.rho=self.all_rho[idx_max]
            self.s=self.all_s[idx_max]
            self.psi=self.fixed_points[idx_max]
            self.stability=self.all_stabilities[idx_max]
            self.physical=False

            

    #=============================================================================================================================
    # Saving and loading
    #=============================================================================================================================  
        
    def save_parameters(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/BP_mu='+str(self.mu)
            if not os.path.exists(path):
                os.makedirs(path)
            path+='/parameters.json'
        parameters={'rule': self.rule, 'chemical potential': self.mu, 'tolerance': self.tol, 'maximum number of iterations': self.max_iter, 'damping parameter': self.damping_parameter}
        with open(path, 'w+') as json_file:
            json.dump(parameters, json_file)

    def save_observables(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/BP_mu='+str(self.mu)
            if not os.path.exists(path):
                os.makedirs(path)
            path+='/observables.json'
        observables={'Messages': self.psi.tolist(), 'Free entropy density': self.phi, 'density': self.rho, 'entropy density': self.s, 'stability': self.stability, 'physical': self.physical, 'stability AT': self.stability_AT, 'linear stability': self.linear_stability, 'non linear stability':self.non_linear_stability, 'eta':self.eta.tolist(), 'phi_SP': self.phi_SP, 'stability SP': self.stability_SP, 'phi_SP_noisy': self.phi_SP_noisy}
        with open(path, 'w+') as json_file:
            json.dump(observables, json_file)
            
    def save_all_fixed_points(self, path=None):
        if self.all_phi is not None:
            if path is None:
                path='results/'+str(self.rule)+'/BP_mu='+str(self.mu)+'/all_fixed_points.csv'
            with open(path, 'w', newline='') as csv_file:
                writer=csv.writer(csv_file)
                writer.writerow(['Messages', 'Free entropy density', 'density', 'entropy density', 'stability', 'physical'])
                for i in range(len(self.all_phi)):
                    writer.writerow([self.fixed_points[i].tolist(), self.all_phi[i], self.all_rho[i], self.all_s[i], self.all_stabilities[i], self.all_physical[i]])
        
            
    def save(self, save_all_fixed_points=True, folder=None, parameters_file='parameters.json', observables_file='observables.json', all_fixed_points_file='all_fixed_points.csv'):
        if folder is None:
            folder='results/'+str(self.rule)+'/BP_mu='+str(self.mu)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.save_parameters(path=folder+'/'+parameters_file)
        self.save_observables(path=folder+'/'+observables_file)
        if save_all_fixed_points:
            self.save_all_fixed_points(path=folder+'/'+all_fixed_points_file)
        
         
    def load_parameters(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/BP_mu='+str(self.mu)+'/parameters.json'
        if os.path.exists(path):
            parameters=json.load(open(path))
            self.rule=parameters['rule']
            self.mu=parameters['chemical potential']
            self.tol=parameters['tolerance']
            self.max_iter=parameters['maximum number of iterations']
            self.damping_parameter=parameters['damping parameter']
            
            self.d=len(self.rule)-1
            self.permutations=np.array(list(itertools.product([0,1], repeat=self.d-1)))
        else:
            print('Warning ! The parameters were not loaded. The file '+path+' does not exist !')
            
    def load_observables(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/BP_mu='+str(self.mu)+'/observables.json'
        if os.path.exists(path):
            observables=json.load(open(path))
            self.psi=np.array(observables['Messages'])
            self.psi_old=self.psi.copy()
            self.phi=observables['Free entropy density']
            self.rho=observables['density']
            self.s=observables['entropy density']
            self.stability=observables['stability']
            self.physical=observables['physical']
            if 'stability AT' in observables:
                self.stability_AT=observables['stability AT']
            if 'linear stability' in observables:
                self.linear_stability=observables['linear stability']
            if 'non linear stability' in observables:
                self.non_linear_stability=observables['non linear stability']
            if 'eta' in observables:
                self.eta=np.array(observables['eta'])
                self.eta_old=self.eta.copy()
            if 'phi_SP' in observables:
                self.phi_SP=observables['phi_SP']
            if 'stability SP' in observables:
                self.stability_SP=observables['stability SP']
            if 'phi_SP_noisy' in observables:
                self.phi_SP_noisy=observables['phi_SP_noisy']
                
        else:
            print('Warning ! The observables were not loaded. The file '+path+' does not exist !')
            
    def load_all_fixed_points(self, path=None):
        if path is None:
            path='results/'+str(self.rule)+'/BP_mu='+str(self.mu)+'/all_fixed_points.csv'
        if os.path.exists(path):
            csv_reader=csv.reader(open(path))
            self.fixed_points=[]
            self.all_phi=[]
            self.all_rho=[]
            self.all_s=[]
            self.all_stabilities=[]
            self.all_physical=[]
            for i, row in enumerate(csv_reader):
                if i!=0:
                    self.fixed_points.append(ast.literal_eval(row[0]))
                    if row[1]=='-inf':
                        self.all_phi.append(-np.inf)
                    elif row[1]=='inf':
                        self.all_phi.append(np.inf)
                    else:
                        self.all_phi.append(ast.literal_eval(row[1]))
                    self.all_rho.append(ast.literal_eval(row[2]))
                    if row[3]=='-inf':
                        self.all_s.append(-np.inf)
                    elif row[3]=='inf':
                        self.all_s.append(np.inf)
                    else:
                        self.all_s.append(ast.literal_eval(row[1]))
                    self.all_stabilities.append(ast.literal_eval(row[4]))
                    self.all_physical.append(ast.literal_eval(row[5]))
            self.fixed_points=np.array(self.fixed_points)
        else:
            print('Warning ! The fixed points were not loaded. The file '+path+' does not exist !')        
        
        
    def load(self, load_all_fixed_points=True, folder=None, parameters_file='parameters.json', observables_file='observables.json', all_fixed_points_file='all_fixed_points.csv'):
        if folder is None:
            folder='results/'+str(self.rule)+'/BP_mu='+str(self.mu)+'/'
        self.load_parameters(path=folder+'/'+parameters_file)
        self.load_observables(path=folder+'/'+observables_file)
        if load_all_fixed_points:
            self.load_all_fixed_points(path=folder+'/'+all_fixed_points_file)
    


    
        
    
    
    