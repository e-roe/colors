import cv2
import numpy as np
import os


'''
   public static RGB whitePatchEbner(int [][] ri, int [][] gi, int [][] bi) {
      RGB result;
      double maxR = 0;
      double maxG = 0;
      double maxB = 0;
      double minR = 255;
      double minG = 255;
      double minB = 255;
      int linhas = ri.length;
      int colunas = ri[0].length;

      double[][] r = new double[linhas][colunas];
      double[][] g = new double[linhas][colunas];
      double[][] b = new double[linhas][colunas];

      int[][] rRes = new int[linhas][colunas];
      int[][] gRes = new int[linhas][colunas];
      int[][] bRes = new int[linhas][colunas];

      double diffR = 0;
      double diffG = 0;
      double diffB = 0;
      int total = linhas * colunas;
      double sortedRed[] = new double[total];
      double sortedGreen[] = new double[total];
      double sortedBlue[] = new double[total];

      for (int l = 0, i = 0; l < linhas; l++) {
         for (int c = 0; c < colunas; c++, i++) {
            r[l][c] = ri[l][c] / 255.0;
            g[l][c] = gi[l][c] / 255.0;
            b[l][c] = bi[l][c] / 255.0;

            if(ri[l][c] > 250 && ri[l][c] > 250 && ri[l][c] > 250) {
               sortedRed[i]   = 0;
               sortedGreen[i] = 0;
               sortedBlue[i]  = 0;
            } else {
               sortedRed[i]   = r[l][c];
               sortedGreen[i] = g[l][c];
               sortedBlue[i]  = b[l][c];
            }
         }
      }

      Arrays.sort(sortedRed);
      Arrays.sort(sortedGreen);
      Arrays.sort(sortedBlue);

//      double pBlack = 0.02;
//      double pWhite = 0.01;

      double pBlack = 0.0;
      double pWhite = 0.04;

      minR = sortedRed[(int)(total * pBlack)];
      maxR = sortedRed[(int)(total * (1 - pWhite))];
      minG = sortedGreen[(int)(total * pBlack)];
      maxG = sortedGreen[(int)(total * (1 - pWhite))];
      minB = sortedBlue[(int)(total * pBlack)];
      maxB = sortedBlue[(int)(total * (1 - pWhite))];
      diffR = maxR - minR;
      diffG = maxG - minG;
      diffB = maxB - minB;

      for(int l = 0; l < linhas; l++) {
         for(int c = 0; c < colunas; c++) {
            r[l][c] = (r[l][c] - minR) / diffR;
            g[l][c] = (g[l][c] - minG) / diffG;
            b[l][c] = (b[l][c] - minB) / diffB;

           // r[l][c] = (r[l][c] - minT) / diffT;
           // g[l][c] = (g[l][c] - minT) / diffT;
           // b[l][c] = (b[l][c] - minT) / diffT;

         }
      }

      for (int l = 0; l < linhas; l++) {
         for (int c = 0; c < colunas; c++) {
            rRes[l][c] =  ImageUtil.clamp((int)(r[l][c] * 255.0));
            gRes[l][c] =  ImageUtil.clamp((int)(g[l][c] * 255.0));
            bRes[l][c] =  ImageUtil.clamp((int)(b[l][c] * 255.0));
         }
      }

      result = new RGB(applyGamma(rRes, 2.2), applyGamma(gRes, 2.2), applyGamma(bRes, 2.));
     // result = new RGB(rRes, gRes, bRes);

      return result;
   }


   public static RGB greyWorldEbner(int [][] ri, int [][] gi, int [][] bi) {
      RGB result;
      double max = 0;
      double min = 255;
      int linhas = ri.length;
      int colunas = ri[0].length;
      double[][] r = new double[linhas][colunas];
      double[][] g = new double[linhas][colunas];
      double[][] b = new double[linhas][colunas];
      int[][] rRes = new int[linhas][colunas];
      int[][] gRes = new int[linhas][colunas];
      int[][] bRes = new int[linhas][colunas];
      int total = linhas * colunas;

      double sorted[] = new double[3 * total];

      for (int l = 0, i = 0; l < linhas; l++) {
         for (int c = 0; c < colunas; c++, i+=3) {
            r[l][c] = ri[l][c] / 255.0;
            g[l][c] = gi[l][c] / 255.0;
            b[l][c] = bi[l][c] / 255.0;

            sorted[i]   = r[l][c];
            sorted[i+1] = g[l][c];
            sorted[i+2] = b[l][c];
         }
      }
      Arrays.sort(sorted);
      double pWhite = 0.02;
      max = sorted[(int)(3 * total * (1 - pWhite))];
      for(int l = 0; l < linhas; l++) {
         for(int c = 0; c < colunas; c++) {
            r[l][c] = r[l][c] / max;
            g[l][c] = g[l][c] / max;
            b[l][c] = b[l][c] / max;
         }
      }

      for (int l = 0; l < linhas; l++) {
         for (int c = 0; c < colunas; c++) {
            rRes[l][c] =  ImageUtil.clamp((int)(r[l][c] * 255.0));
            gRes[l][c] =  ImageUtil.clamp((int)(g[l][c] * 255.0));
            bRes[l][c] =  ImageUtil.clamp((int)(b[l][c] * 255.0));
         }
      }

      result = new RGB(rRes, gRes, bRes);

      return result;
   }

'''
def max_white(nimg):
    if nimg.dtype==np.uint8:
        brightest=float(2**8)
    elif nimg.dtype==np.uint16:
        brightest=float(2**16)
    elif nimg.dtype==np.uint32:
        brightest=float(2**32)
    else:
        brightest=float(2**8)
    nimg = nimg.transpose(2, 0, 1)
    nimg = nimg.astype(np.int32)
    nimg[0] = np.minimum(nimg[0] * (brightest/float(nimg[0].max())),255)
    nimg[1] = np.minimum(nimg[1] * (brightest/float(nimg[1].max())),255)
    nimg[2] = np.minimum(nimg[2] * (brightest/float(nimg[2].max())),255)

    return nimg.transpose(1, 2, 0).astype(np.uint8)


def retinex(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = nimg[1].max()
    nimg[0] = np.minimum(nimg[0]*(mu_g/float(nimg[0].max())),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


def grey_world2(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    #nimg[1] = np.minimum(nimg[1] * (mu_g / np.average(nimg[1])), 255)
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)

    return  nimg.transpose(1, 2, 0).astype(np.uint8)

import sys

def maxw(image, name):
    imgc = image / 255.
    mea = np.mean(imgc)

    pWhite = 0.04

    r = imgc[:, :, 2]
    g = imgc[:, :, 1]
    b = imgc[:, :, 0]

    rs = sorted(r.ravel())
    gs = sorted(g.ravel())
    bs = sorted(b.ravel())

    total = len(rs)

    max_index = int(total * (1. - pWhite))
    print(bs[max_index], bs[max_index-1])
    maxr = rs[max_index]
    maxg = gs[max_index]
    maxb = bs[max_index]
    minr = rs[0]
    ming = gs[0]
    minb = bs[0]
    print(maxr, maxg, maxb)
    max = np.max([maxr, maxg, maxb])
    min = np.min([minr, ming, minb])
    print('>>>>', max, min)

    r = (r-minr / (maxr-minr))
    g = (g-ming / (maxg-ming))
    b = (b-minb / (maxb-minb))

    imgc[:, :, 2] = r
    imgc[:, :, 1] = g
    imgc[:, :, 0] = b

    return imgc

def grey_world(image, name):
    imgc = image / 255.
    mea = np.mean(imgc)

    pWhite = 0.05

    r = imgc[:, :, 2] / np.mean(imgc[:, :, 2])
    g = imgc[:, :, 1] / np.mean(imgc[:, :, 1])
    b = imgc[:, :, 0] / np.mean(imgc[:, :, 0])

    rs = sorted(r.ravel())
    gs = sorted(g.ravel())
    bs = sorted(b.ravel())

    total = len(rs)

    max_index = int(total * (1. - pWhite))
    print(bs[max_index], bs[max_index-1])

    print(total, int(total * (1. - pWhite)))
    maxr = rs[max_index]
    maxg = gs[max_index]
    maxb = bs[max_index]
    print(maxr, maxg, maxb)
    max = np.max([maxr, maxg, maxb])
    print(max)

    r = (r / maxr)
    g = (g / maxg)
    b = (b / maxb)

    imgc[:, :, 2] = r
    imgc[:, :, 1] = g
    imgc[:, :, 0] = b

    return imgc
    image[:, :, 2] = np.clip(r * 255, 0, 255)
    image[:, :, 1] = np.clip(g * 255, 0, 255)
    image[:, :, 0] = np.clip(b * 255, 0, 255)

    return image

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def apply(image, name, factor):

    res_path = os.path.join('D:\\Roe\\Medium\\paper_to\\CConstancy\\constancy_project\\res', name)
    os.makedirs(res_path, exist_ok=True)

    image = cv2.resize(image, (0, 0), fx=factor, fy=factor)
    mm = maxw(image, name)

    #mm = white_balance(image.copy())

    gg1 = grey_world(image.copy(), name)
    cv2.imshow('grey', gg1)

    gg2 = grey_world2(image.copy())
    cv2.imshow('grey2', gg2)

    cv2.imshow('orig', image)
    cv2.imshow('max', mm)

    wbg = cv2.xphoto.createGrayworldWB()
    wbg.setSaturationThreshold(0.1499)
    imageg = wbg.balanceWhite(image)

    wbs = cv2.xphoto.createSimpleWB()
    images = wbs.balanceWhite(image)

    wbb = cv2.xphoto.createLearningBasedWB()
    imageb = wbb.balanceWhite(image)
    cv2.imshow('wbb', imageg)

    ret = retinex(image)
    cv2.imwrite(os.path.join(res_path, 'original.png'), image)
    cv2.imwrite(os.path.join(res_path, 'grey_world.png'), gg1 * 255)
    cv2.imwrite(os.path.join(res_path, 'cv2_simple_wb.png'), images)
    cv2.imwrite(os.path.join(res_path, 'cv2_wb.png'), imageb)
    cv2.imwrite(os.path.join(res_path, 'ret.png'), ret)
    cv2.imwrite(os.path.join(res_path, 'cv2_grey_world.png'), imageg)
    cv2.imwrite(os.path.join(res_path, 'wb.png'), mm * 255)
    # cv2.imwrite(os.path.join(res_path, '.png'), )


    cv2.waitKey(0)
    sys.exit()

    gg = grey_world2(image.copy())

    max = max_white(image.copy())
    cv2.imshow('original', image)
    cv2.imshow('grey_world2', gg)
    cv2.imshow('grey_world', gg1)
    cv2.imshow('max', max)
    cv2.waitKey(0)

if __name__ == '__main__':
    file = 'Wb_girl_tungsten.jpg'
    #file = 'Wb_girl_warm.jpg'
    #file = 'Wb_girl_cloudy.jpg'
    file = 'belly.jpg'
    image = cv2.imread(os.path.join('D:\\Roe\\Medium\\paper_to\\CConstancy\\constancy_project\\imgs', file))
    apply(image, file[:-4], 1)
