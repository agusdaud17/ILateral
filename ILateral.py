#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 05:56:32 2022

@author: agus
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from shapely.geometry import Polygon
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# from plotly.subplots import make_subplots

dec = 3

def atRest_NC(phi=30):
    phi = np.radians(phi)
    K0 = 1 - np.sin(phi)
    return np.round(K0,dec)

def atRest_OC(phi=30,OCR=2):
    phi = np.radians(phi)
    K0_OCR = (1 - np.sin(phi)) * OCR**np.sin(phi)
    return np.round(K0_OCR,dec)

def active_R(phi=30):
    Ka = np.tan(np.radians(45-phi/2))**2
    return np.round(Ka,dec)

def passive_R(phi=30):
    Kp = np.tan(np.radians(45+phi/2))**2
    return np.round(Kp,dec)

def genActive_R(phi=30,beta=0,theta=0):
    # beta = inklined backfill
    # theta = inklined backs wall
    # xi = Pa inclined at angle xi

    psi = (np.arcsin(beta / phi)*180/np.pi) - beta + (2*theta)

    psi = np.radians(psi)
    phi = np.radians(phi)
    beta = np.radians(beta)
    theta = np.radians(theta)

    Ka = ( np.cos(beta-theta) * np.sqrt(1 + (np.sin(phi)**2) - (2*np.sin(phi)*np.cos(psi))) ) / ((np.cos(theta)**2) * (np.cos(beta) + np.sqrt((np.sin(phi)**2) - (np.sin(beta)**2))) )
    return np.round(Ka,dec)

def genPassive_R(phi=30,beta=0,theta=0):
    # beta = inklined backfill
    # theta = inklined backs wall
    # xi = Pa inclined at angle xi

    psi = (np.arcsin(beta / phi)*180/np.pi) + beta - (2*theta)

    psi = np.radians(psi)
    phi = np.radians(phi)
    beta = np.radians(beta)
    theta = np.radians(theta)
    # xi = np.arctan( (np.sin(np.radians((phi)))*np.sin(np.radians((psi)))) / (1 - (np.sin(np.radians(phi))*np.cos(np.radians(psi)))) )*180/np.pi
    Kp = ( np.cos(beta-theta) * np.sqrt(1 + (np.sin(phi)**2) + (2*np.sin(phi)*np.cos(psi))) ) / ((np.cos(theta)**2) * (np.cos(beta) - np.sqrt((np.sin(phi)**2) - (np.sin(beta)**2))) )

    return np.round(Kp,dec)

def active_C(phi=30,beta=0,theta=0,delta=-1):
    # vertical retaining wall beta = 90 deg
    # horizontal backfill beta = 0 deg
    # wall friction angle, delta = phi/3
    if delta == -1:
        delta = phi/3
    phi = np.radians(phi)
    beta = np.radians(beta)
    theta = np.radians(theta)
    delta = np.radians(delta)
    Ka = np.cos(phi-theta)**2 / ( (np.cos(theta)**2) * np.cos(theta+delta) * (1 + np.sqrt( (np.sin(phi+delta) * np.sin(phi-beta)) / (np.cos(theta-delta) * np.cos(theta+beta))) )**2)
    return np.round(Ka,dec)

def passive_C(phi=30,beta=0,theta=0,delta=-1):
    # vertical retaining wall beta = 90 deg
    # horizontal backfill beta = 0 deg
    # wall friction angle, delta = phi/3
    if delta == -1:
        delta = phi/3
    phi = np.radians(phi)
    beta = np.radians(beta)
    theta = np.radians(theta)
    delta = np.radians(delta)
    Kp = np.cos(phi+theta)**2 / ( (np.cos(theta)**2) * np.cos(theta-delta) * (1 - np.sqrt( (np.sin(phi+delta) * np.sin(phi+beta)) / (np.cos(theta-delta) * np.cos(theta-beta))) )**2)
    return np.round(Kp,dec)

def active_MO(phi=30,kh=0.1,kv=0,alpha=0,beta=90,delta=0):
    theta = np.arctan(kh / (1-kv)) * 180/np.pi
    theta = np.radians(theta)
    phi = np.radians(phi)
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    delta = np.radians(delta)
    Kae = np.sin(phi+beta-theta)**2 / ( np.cos(theta) * (np.sin(beta)**2) * np.sin(beta-theta-delta) * (1 + np.sqrt( (np.sin(phi+delta) * np.sin(phi-theta-alpha)) / (np.sin(beta-delta-theta) * np.sin(alpha+beta))) )**2)
    return np.round(Kae,dec)

class Turap:
    """docstring for turap."""

    def __init__(self, name="T1",H0=3,d0=0,z_uf=99,z_ub=99,Rd=1.2,FS_phi=1.25,delta_Kp=1/2,delta_Ka=2/3,z_anchor=99):
        self.name = name
        self.z_anchor = z_anchor

        if self.z_anchor < 99:
            if d0 == 0:
                self.d0 = H0/3
            else:
                self.d0 = d0
            self.Rd = 1.01
        else:
            if d0 == 0:
                self.d0 = H0 #trial embeded lenght
            else:
                self.d0 = d0
            self.Rd = Rd

        self.H0 = H0
        self.FS_phi = FS_phi
        self.zu_front = z_uf
        self.zu_back = z_ub
        self.delta_Kp = delta_Kp
        self.delta_Ka = delta_Ka

    def insert_row(self,z,table):
        if z < table["z"].iloc[-1]:
            index = table.loc[table['z'] >= z].index[0]

            if table.loc[index,'z'] == z:
                table2 = table
            else:
                table.loc[index-0.5] = table.loc[index]
                table.loc[index-0.5,'z'] = z
                table.sort_values(by=['z'])
                table1 = table.sort_index()
                table2 = table1.reset_index(drop=True)
        else:
            table2 = table
        return table2

    def insert_interv(self,z,table):
        index = table.loc[table['z'] >= z].index[0]
        table.loc[index-0.5] = table.loc[index]
        table.loc[index-0.5,'z'] = z
        table.sort_values(by=['z'])
        table1 = table.sort_index()
        table2 = table1.reset_index(drop=True)
        return table2

    def load_excel(self,name):
        self.excel_name = name
        self.raw_df = pd.read_excel(name)
        self.df = pd.read_excel(name)

    def add_depth(self):
        z_insert = [self.H0,
                    self.H0+self.d0,
                    self.H0+(self.Rd*self.d0),
                    self.zu_front,
                    self.zu_back]

        for z in z_insert:
            self.df = self.insert_row(z,self.df)

        # list_z = np.arange(0,self.H0+(self.Rd*self.d0),0.5)
        # for z in list_z:
        #     self.df = self.insert_row(z,self.df)

        # insert null params
        self.df["Ka"] = 0
        self.df["Kp"] = 0
        self.df["po_front"] = 0
        self.df["u_front"] = 0
        self.df["po_back"] = 0
        self.df["u_back"] = 0

    def find_moment_arm(self,list_x,list_y):
        list_coord = list(zip(list_x,list_y))
        block_A = np.column_stack((list_x,list_y))
        block_B = np.flipud(block_A)
        block_B[:,0] = block_B[:,0]*0
        block_A = np.column_stack((list_x,list_y))
        block = np.vstack((block_A,block_B))
        P = Polygon(block).area
        z_o = (self.H0+self.d0) - Polygon(block).centroid.coords.xy[1].tolist()[0]
        return P, z_o

    def solve_params(self):
        for i in range(1,len(self.df['z'])):
            z_prev = self.df.loc[i-1,"z"]
            z_curr = self.df.loc[i,"z"]
            g = self.df.loc[i,"g"]
            phi = self.df.loc[i,"phi"] / self.FS_phi
            delta = self.df.loc[i,"delta"] / self.FS_phi

            Kp = passive_C(phi=phi,delta=phi*self.delta_Kp,beta=self.beta)
            Ka = active_C(phi=phi,delta=phi*self.delta_Ka)

            self.df.loc[i,"Kp"] = Kp
            self.df.loc[i,"Ka"] = Ka

            if z_curr > self.zu_front:
                self.df.loc[i,"u_front"] = (z_curr - self.zu_front)*9.81

            if z_curr > self.zu_back:
                self.df.loc[i,"u_back"] = (z_curr - self.zu_back)*9.81

            if z_curr == 0:
                pass
            elif z_curr <= self.H0:
                h_back = z_curr - z_prev
                po_back = (g*h_back) + self.df.loc[i-1,"po_back"]
                self.df.loc[i,"po_back"] = po_back
            elif z_curr > self.H0:
                h_front = z_curr - z_prev
                po_front = (g*h_front) + self.df.loc[i-1,"po_front"]
                self.df.loc[i,"po_front"] =  po_front

                h_back = z_curr - z_prev
                po_back = (g*h_back) + self.df.loc[i-1,"po_back"]
                self.df.loc[i,"po_back"] = po_back

        self.df["po_ef_front"] = (self.df["po_front"] - self.df["u_front"])
        self.df["po_ef_back"] = (self.df["po_back"] - self.df["u_back"])

        # duplicate depth
        for z in self.df['z']:
            if z == 0:
                pass
            else:
                self.df = self.insert_interv(z,self.df)

        # revise Ka and Kp value
        for i in range(len(self.df["z"])-1):
            if i == 0:
                pass
            elif (i % 2) == 0:
                self.df.loc[i,"Ka"] = self.df.loc[i+1,"Ka"]
                self.df.loc[i,"Kp"] = self.df.loc[i+1,"Kp"]

        for i in range(len(self.df["po_ef_front"])):
            if self.df.loc[i,"po_ef_front"] < 0:
                self.df.loc[i,"po_ef_front"] = 0
            if self.z_anchor >= 99:
                if self.df.loc[i,"z"] >= self.H0+(self.Rd*self.d0):
                    Kp = self.df.loc[i-1,"Kp"]
                    Ka = self.df.loc[i-1,"Ka"]
                    self.df.loc[i-1,"Ka"] = Kp
                    self.df.loc[i-1,"Kp"] = Ka
            else:
                pass

        self.df['pp_front'] = -self.df['Kp']*self.df['po_ef_front'] - self.df['u_front'] #- (2*self.df['c']*(self.df['Kp']**(0.5)))
        if self.qs == 0:
            self.df['pa_back'] = self.df['Ka']*self.df['po_ef_back'] + self.df['u_back'] - (2*self.df['c']*(self.df['Ka']**(0.5)))
        else:
            self.df['pa_back'] = self.df['Ka']*self.df['po_ef_back'] + self.df['u_back'] - (2*self.df['c']*(self.df['Ka']**(0.5))) + self.df['Ka']*self.qs

        # delete row more than embedded length
        self.df = self.df.drop(self.df[self.df['z'] > self.H0+(self.Rd*self.d0)].index)

        # Resultant earth pressure
        self.df["pr"] = self.df["pa_back"] + self.df["pp_front"]

    def solve_earthPressure(self):
        i_exc = self.df.loc[self.df['z'] == self.H0].index[0]
        i_0 = self.df.loc[self.df['z'] >= self.H0+self.d0].index[0]
        i_end = self.df.loc[self.df['z'] >= self.H0+(self.Rd*self.d0)].index[0]

        # Force above point O
        list_x = self.df.loc[i_exc:i_0,'pp_front'].to_numpy()
        list_y = self.df.loc[i_exc:i_0,'z'].to_numpy()
        self.Ppx, self.zp = self.find_moment_arm(list_x,list_y)

        list_x_back = self.df.loc[:i_0,'pa_back'].to_numpy()
        list_z_back = self.df.loc[:i_0,'z'].to_numpy()
        self.Pax, self.za = self.find_moment_arm(list_x_back,list_z_back)

        self.R = self.Ppx - self.Pax

        #Force bellow point O
        list_x = self.df.loc[i_0:i_end,'pp_front'].to_numpy()
        list_y = self.df.loc[i_0:i_end,'z'].to_numpy()
        self.Pax_end, self.za_end = self.find_moment_arm(list_x,list_y)

        list_x = self.df.loc[i_0:i_end,'pa_back'].to_numpy()
        list_y = self.df.loc[i_0:i_end,'z'].to_numpy()
        self.Ppx_end, self.zp_end = self.find_moment_arm(list_x,list_y)
        self.R_end = self.Ppx_end - self.Pax_end

        # calc moment max
        list_x = self.df.loc[:i_0,'pr'].to_numpy()
        list_y = self.df.loc[:i_0,'z'].to_numpy()
        self.Pmax, self.z_max = self.find_moment_arm(list_x,list_y)

        if self.z_anchor >= 99:
            self.Mp = (self.Ppx*self.zp)
            self.Ma = (self.Pax*self.za)
            self.FS = np.round(self.Mp/self.Ma,2)
            self.zr = (self.Mp-self.Ma)/(self.Ppx-self.Pax)
        else:
            zp = self.H0 + self.d0 - self.zp - self.z_anchor
            za = self.H0 + self.d0 - self.za - self.z_anchor
            self.Mp = (self.Ppx*zp)
            self.Ma = (self.Pax*za)
            self.FS = np.round(self.Mp/self.Ma,2)
            self.F_anchor = np.round(self.Ppx - self.Pax,2)

    def solve(self,FSmin=0.99,qs=0,beta=0):
        self.qs = qs
        self.beta = beta
        while True:
            self.add_depth()
            self.solve_params()
            self.solve_earthPressure()
            if FSmin < self.FS:
                self.L_emb = np.round(self.Rd*self.d0,2)
                break
                # if self.R <= self.R_end:
                #     self.L_emb = np.round(self.Rd*self.d0,2)
                #     break
            else:
                self.load_excel(self.excel_name)
                self.d0 += 0.1

    def plot_diagram(self):
        fig = go.Figure(layout = go.Layout({"showlegend": False}))
        df = self.df
        P_max = np.max(np.abs(df["pr"]))
        fig.add_trace(go.Scatter(x=[0,0], y=[0,self.H0+self.L_emb],mode='lines',line=dict(color="#000",width= 4)))
        fig.add_trace(go.Scatter(x=df["pr"], y=df["z"],mode='lines',line=dict(color="#33cc33"),name='Pressure',fill='tonexty'))
        fig.update_layout(width=550,height=550, title="Lateral Earth Pressure Diagram", xaxis_title="Lateral Pressure (kPa/m)",yaxis_title="Depth (m)")
        fig.add_trace(go.Scatter(x=[0,1.5*P_max], y=[0,-np.tan(np.radians(self.beta))*1.5],mode='lines',line=dict(color="#000")))
        fig.add_trace(go.Scatter(x=[0,-P_max*1.5], y=[self.H0,self.H0],mode='lines',line=dict(color="#000")))

        if self.zu_back < 99:
            fig.add_trace(go.Scatter(x=[0,1.5*P_max], y=[self.zu_back,self.zu_back],mode='lines',line=dict(color="#6495ED",dash = 'dash')))
        if self.zu_front < 99:
            fig.add_trace(go.Scatter(x=[0,-1.5*P_max], y=[self.zu_front,self.zu_front],mode='lines',line=dict(color="#6495ED",dash = 'dash')))

        fig['layout']['yaxis']['autorange'] = "reversed"
        fig.show()

    def solve_internal_force(self,dz=0.01):
        z_0 = self.H0+self.d0
        list_z = np.arange(0,z_0,dz)
        xp = self.df["z"]
        fp = self.df["pr"]

        list_p = np.interp(list_z, xp, fp)

        list_z_new = []
        list_shear = []
        list_moment = []

        if self.z_anchor >= 99:
            for i, z in enumerate(list_z):
                if i ==0:
                    v = 0
                    m = 0
                else:
                    v =np.round(np.sum(list_p[:i]*dz),2)
                    m = np.round(np.sum(np.abs(list_z[:i] - list_z[i]) * (list_p[:i]*dz) * 0.5),2)
                list_z_new.append(z)
                list_shear.append(v)
                list_moment.append(m)
        else:
            for i, z in enumerate(list_z):
                if i == 0:
                    v = 0
                    m = 0
                elif z <= self.z_anchor:
                    v = np.round(np.sum(list_p[:i]*dz),2)
                    m = np.round(np.sum(np.abs(list_z[:i] - list_z[i]) * (list_p[:i]*dz) * 0.5),2)
                else:
                    v = np.round(np.sum(list_p[:i]*dz) + self.F_anchor,2)
                    m = np.round(np.sum(np.abs(list_z[:i] - list_z[i]) * (list_p[:i]*dz) * 0.5) + (0.5*self.F_anchor*(z-self.z_anchor)),2)

                list_z_new.append(z)
                list_shear.append(v)
                list_moment.append(m)

        self.Mmax = np.max(np.abs(list_moment))
        self.Vmax = np.max(np.abs(list_shear))
        self.internal_force_table = pd.DataFrame(columns=["Depth (m)","Shear (kN)","Moment (kNm)"])
        self.internal_force_table["Depth (m)"] = list_z_new
        self.internal_force_table["Shear (kN)"] = list_shear
        self.internal_force_table["Moment (kNm)"] = list_moment

        if self.z_anchor < 99:
            for i, z in enumerate(self.internal_force_table["Depth (m)"]):
                if z > self.H0:
                    if self.internal_force_table.loc[i,"Moment (kNm)"] > 0:
                        self.internal_force_table.loc[i,"Moment (kNm)"] = 0
        else:
            for i, m in enumerate(self.internal_force_table["Moment (kNm)"]):
                if m < 0:
                    self.internal_force_table.loc[i,"Moment (kNm)"] = 0
                    self.internal_force_table.loc[i,"Shear (kN)"] = 0

    def solve_deflection(self,E=200000000,I=0.0001814):
        EI = E*I
        dz = 0.01

        df_m = self.internal_force_table["Moment (kNm)"]

        self.internal_force_table["dy"] = (df_m/EI)*(dz**2)
        i_last = len(df_m)
        if self.z_anchor >= 99:
            for i, dy in enumerate(self.internal_force_table["dy"]):
                y = np.sum(self.internal_force_table.loc[i:,"dy"])
                self.internal_force_table.loc[i,"Deflection (cm)"] = -y*100
        else:
            for i, dy in enumerate(self.internal_force_table["dy"]):
                z = self.internal_force_table.loc[i,"Depth (m)"]
                index = self.internal_force_table.loc[self.internal_force_table["Depth (m)"] >= self.z_anchor].index[0]
                if z <= self.z_anchor:
                    y = np.sum(self.internal_force_table.loc[i:index,"dy"])
                else:
                    y = np.sum(self.internal_force_table.loc[i:,"dy"]) * (z-self.z_anchor)/(self.H0+self.d0 - self.z_anchor)
                self.internal_force_table.loc[i,"Deflection (cm)"] = y*100

    def plot_bending_moment(self):
        fig = go.Figure(layout = go.Layout({"showlegend": False}))
        df = self.internal_force_table
        fig.add_trace(go.Scatter(x=[0,0], y=[0,self.H0+self.L_emb],mode='lines',line=dict(color="#000",width= 4)))
        fig.add_trace(go.Scatter(x=df["Moment (kNm)"], y=df["Depth (m)"],mode='lines',line=dict(color="#FF5733"),name='Moment',fill='tonexty'))
        fig.update_layout(width=550,height=550, title="Bending Moment Diagram", xaxis_title="Bending moment (kNm)",yaxis_title="Depth (m)")
        fig.add_trace(go.Scatter(x=[0,1.5*self.Mmax], y=[0,-np.tan(np.radians(self.beta))*1.5],mode='lines',line=dict(color="#000")))
        fig.add_trace(go.Scatter(x=[0,-self.Mmax*1.5], y=[self.H0,self.H0],mode='lines',line=dict(color="#000")))

        if self.zu_back < 99:
            fig.add_trace(go.Scatter(x=[0,1.5*self.Mmax], y=[self.zu_back,self.zu_back],mode='lines',line=dict(color="#6495ED",dash = 'dash')))
        if self.zu_front < 99:
            fig.add_trace(go.Scatter(x=[0,-1.5*self.Mmax], y=[self.zu_front,self.zu_front],mode='lines',line=dict(color="#6495ED",dash = 'dash')))

        fig['layout']['yaxis']['autorange'] = "reversed"
        fig.show()

    def plot_shear_force(self):
        fig = go.Figure(layout = go.Layout({"showlegend": False}))
        df = self.internal_force_table
        Vmax = np.max(np.abs(df["Shear (kN)"]))
        fig.add_trace(go.Scatter(x=[0,0], y=[0,self.H0+self.L_emb],mode='lines',line=dict(color="#000",width= 4)))
        fig.add_trace(go.Scatter(x=df["Shear (kN)"], y=df["Depth (m)"],mode='lines',line=dict(color="#0080ff"),name='Shear',fill='tonexty'))
        fig.update_layout(width=550,height=550, title="Shear Force Diagram", xaxis_title="Shear force (kN)",yaxis_title="Depth (m)")
        fig.add_trace(go.Scatter(x=[0,1.5*Vmax], y=[0,-np.tan(np.radians(self.beta))*1.5],mode='lines',line=dict(color="#000")))
        fig.add_trace(go.Scatter(x=[0,-Vmax*1.5], y=[self.H0,self.H0],mode='lines',line=dict(color="#000")))

        if self.zu_back < 99:
            fig.add_trace(go.Scatter(x=[0,1.5*Vmax], y=[self.zu_back,self.zu_back],mode='lines',line=dict(color="#6495ED",dash = 'dash')))
        if self.zu_front < 99:
            fig.add_trace(go.Scatter(x=[0,-1.5*Vmax], y=[self.zu_front,self.zu_front],mode='lines',line=dict(color="#6495ED",dash = 'dash')))

        fig['layout']['yaxis']['autorange'] = "reversed"
        fig.show()

    def plot_deformation(self):
        fig = go.Figure(layout = go.Layout({"showlegend": False}))
        df = self.internal_force_table
        Dmax = np.max(np.abs(df["Deflection (cm)"]))
        fig.add_trace(go.Scatter(x=[0,0], y=[0,self.H0+self.L_emb],mode='lines',line=dict(color="#888",width= 3,dash = 'dash')))
        fig.add_trace(go.Scatter(x=df["Deflection (cm)"], y=df["Depth (m)"],mode='lines',line=dict(color="#000",width= 4),name='deflection'))
        fig.update_layout(width=550,height=550, title="Deformation Diagram", xaxis_title="Deflection (cm)",yaxis_title="Depth (m)")
        fig.add_trace(go.Scatter(x=[0,5*Dmax], y=[0,-np.tan(np.radians(self.beta))*1.5],mode='lines',line=dict(color="#000")))
        fig.add_trace(go.Scatter(x=[0,-Dmax*5], y=[self.H0,self.H0],mode='lines',line=dict(color="#000")))

        if self.zu_back < 99:
            fig.add_trace(go.Scatter(x=[0,5*Dmax], y=[self.zu_back,self.zu_back],mode='lines',line=dict(color="#6495ED",dash = 'dash')))
        if self.zu_front < 99:
            fig.add_trace(go.Scatter(x=[0,-5*Dmax], y=[self.zu_front,self.zu_front],mode='lines',line=dict(color="#6495ED",dash = 'dash')))

        fig['layout']['yaxis']['autorange'] = "reversed"
        fig.show()
