#
#    Copyright 2021-2022 konawasabi
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

'''
'''

import numpy as np
import pandas as pd
import pyproj
import folium

def rotate(tau1):
    '''２次元回転行列を返す。

    tau1: 回転角度 [rad]
    '''
    return np.array([[np.cos(tau1), -np.sin(tau1)], [np.sin(tau1),  np.cos(tau1)]])

def minimumdist(track,p):
    '''二次元曲線trackについて、座標pから最も近い曲線上の点を求める。
    trackの点間は線形補間される。
    
    Args:
        track (ndarray):
            np.array([x0,y0],[x1,y1],...,[xn,yn])
        p (ndarray):
            np.array([xp,yp])

    Returns:
        float
            mindist: 曲線交点との距離
        ndarray
            crosspt: 曲線との交点座標
        int
            min_ix: track中で最もpに近い点のindex
        int
            second_min_ix: 最もpに近い点が含まれる区間の他端index。直交点が見つからない場合は-1。
    '''
    
    dist = (track - p)**2
    min_ix = np.argmin(np.sqrt(dist[:,0]+dist[:,1]))

    def distance(source, index):
        return np.sqrt(source[index][0]+source[index][1])

    if min_ix < len(dist)-1 and min_ix > 0:
        around_min = [distance(dist,min_ix-1),distance(dist,min_ix),distance(dist,min_ix+1)]

        second_min_ix = min_ix + 1

        lenAC = np.sqrt((track[second_min_ix][0] - track[min_ix][0])**2+(track[second_min_ix][1] - track[min_ix][1])**2)
        n = np.array([(track[second_min_ix][0] - track[min_ix][0])/lenAC,(track[second_min_ix][1] - track[min_ix][1])/lenAC])       
        a =np.array([track[min_ix][0],track[min_ix][1]])

        alpha = -np.dot(a-p,n)    
        mindist=(np.linalg.norm(a-p+alpha*n))
        crosspt=(a+alpha*n)

        # 求めたcrossptがtrack[min_ix] ~ track[second_min_ix]の間にない場合
        if crosspt[0]< track[min_ix][0] or crosspt[0]> track[second_min_ix][0]:
            second_min_ix = min_ix - 1

            lenAC = np.sqrt((track[min_ix][0] - track[second_min_ix][0])**2+(track[min_ix][1] - track[second_min_ix][1])**2)
            n = np.array([(track[min_ix][0] - track[second_min_ix][0])/lenAC,(track[min_ix][1] - track[second_min_ix][1])/lenAC])
            a = np.array([track[second_min_ix][0],track[second_min_ix][1]])

            alpha = -np.dot(a-p,n)    
            mindist = (np.linalg.norm(a-p+alpha*n))
            crosspt = (a+alpha*n)
    else:
        mindist=(distance(dist,min_ix))
        crosspt=(track[min_ix])
        second_min_ix = -1
        #print(p,mindist,crosspt,min_ix,second_min_ix)

    return mindist, crosspt, min_ix, second_min_ix

def cross_kilopost(track, result):
    '''minimumdistで求めたtrack上の最近傍点について、対応する距離程を求める。
    
    Args:
        track (ndarray):
            np.array([x0,y0],[x1,y1],...,[xn,yn])
        result (list):
            minimumdistの出力

    Return:
        float: 最近傍点の距離程。track端点が最近傍点の場合はNone
    '''
    if result[3]>0:
        subdist = np.sqrt((track[result[2]][1]-result[1][0])**2+(track[result[2]][2]-result[1][1])**2)
        if result[3] < result[2]:
            subdist *= -1
        kilopost = track[result[2]][0] + subdist
    else:
        kilopost = track[result[2]][0]
    return kilopost

def cross_normal(position, track):
    '''曲線trackの法線のうち、positionを通過するものを求める

    Args:
         ndarray
            position: np.array([x,y])
         ndarray
            track: np.array([[x0,y0],[x1,y1],...,[xn,yn]])

    Return:
    '''
    return None
def angle_twov(phiA, phiB):
    ''' ベクトルA (方位角phiA)からベクトルB(方位角phiB)への方位角変化を求める
    '''
    eA = np.array([np.cos(phiA),np.sin(phiA)])
    eB = np.array([np.cos(phiB),np.sin(phiB)])
    return np.arccos(np.dot(eA,eB))*np.sign(np.cross(eA,eB))

def interpolate_with_dist(track, element, cp_dist):
    def interpolate(data,ix,typ,cp_dist,base=0):
        return (data[:,typ][ix+1]-data[:,typ][ix])/(data[:,base][ix+1]-data[:,base][ix])*(cp_dist-data[:,base][ix])+data[:,typ][ix]
    min_ix = np.argmin(np.abs(track[:,0] - cp_dist))

    if min_ix > 0 and min_ix < len(track)-1:
        aroundzero = track[min_ix-1:min_ix+2]
        sign_dist = np.sign(aroundzero[:,0] - cp_dist)
        if sign_dist[0] != sign_dist[1]:
            result = interpolate(aroundzero,0,element,cp_dist)
        else:
            result = interpolate(aroundzero,1,element,cp_dist)
        #result = track[pos_ix][element]
    else:
        result = track[min_ix][element]
    return result

def calc_pl2xy(phi_deg, lambda_deg, phi0_deg, lambda0_deg):
    #phi_deg=위도 lambda_deg경도
    """ 위도 경도를 평면 직각 좌표로 변환
     - input:
         (phi_deg, lambda_deg) : 변환하고 싶은 위도·경도[도](분·초가 아닌 소수인 것에 주의)
         (phi0_deg, lambda0_deg): 평면 직각 좌표계 원점의 위도·경도[도](분·초가 아닌 소수인 것에 주의)
     - output:
         x: 변환 후 평면 직각 좌표[m]
         y: 변환 후 평면 직각 좌표[m]
         * x축 정방향은 북, y축 정방향은 동쪽을 가리키는 것에 주의
        
         원전
         https://qiita.com/sw1227/items/e7a590994ad7dcd0e8ab
     """
    
    p1_type = pyproj.CRS.from_epsg(4326)
    p2_type = pyproj.CRS.from_epsg(5186)
    transformer = pyproj.Transformer.from_crs(p1_type, p2_type, always_xy=True)
    y,x = transformer.transform(lambda_deg,phi_deg)
    #x위도、y경도
    return x, y  # [m]
    
def calc_xy2pl(x, y, phi0_deg, lambda0_deg):
    """평면 직각 좌표를 위도 경도로 변환
     - input:
         (x, y): 변환하려는 x, y 좌표[m]
         (phi0_deg, lambda0_deg): 평면 직각 좌표계 원점의 위도·경도[도](분·초가 아닌 소수인 것에 주의)
         * x축 정방향은 북, y축 정방향은 동쪽을 가리키는 것에 주의
     - output:
         latitude: 위도[도]
         longitude: 경도[도]
         * 소수점 이하는 분·초가 아닌 것에 주의
        
         원전
         https://qiita.com/sw1227/items/e7a590994ad7dcd0e8ab
     """
    p1_type = pyproj.CRS.from_epsg(5186)
    p2_type = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(p1_type, p2_type, always_xy=False)
    latitude, longitude = transformer.transform(x, y)
    return latitude, longitude # [deg]

def long2px(l, z):
    '''経度をマップタイルのピクセル座標に変換する
    - input:
        l: 変換したい経度(longitude) [deg]
        z: ズームレベル (0-18)
    - output:
        x: ピクセル座標x成分 (256で割るとタイル座標になる)
    '''
    return 2**(z+7)*(l/180+1)

def lat2py(l, z, L=85.05112878):
    '''緯度をマップタイルのピクセル座標に変換する
    - input:
        l: 変換したい緯度(latitude) [deg]
        z: ズームレベル (0-18)
        L: 緯度の上限値 [deg]
    - output:
        y: ピクセル座標y成分 (256で割るとタイル座標になる)
    '''
    return (2**(z+7)/np.pi)*(-np.arctanh(np.sin(np.pi/180*l))+np.arctanh(np.sin(np.pi/180*L)))

def px2long(x, z):
    '''ピクセル座標を経度に変換する
    - input:
        x: 変換したいピクセル座標x成分 
        z: ズームレベル (0-18)
    - output:
        l: 経度 [deg]
    '''
    return 180*(x/(2**(z+7))-1)

def py2lat(y, z, L=85.05112878):
    '''ピクセル座標を経度に変換する
    - input:
        y: 変換したいピクセル座標y成分
        z: ズームレベル (0-18)
        L: 緯度の上限値 [deg]
    - output:
        l: 緯度 [deg]
    '''
    return 180/np.pi*(np.arcsin(np.tanh(-np.pi/(2**(z+7))*y+np.arctanh(np.sin(np.pi/180*L)))))