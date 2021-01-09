import dlib
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
from scipy import optimize
from dlib import rectangle
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

from scipy import optimize


from tkinter import filedialog
import os

def getNormal(triangle):
    a = triangle[:, 0]
    b = triangle[:, 1]
    c = triangle[:, 2]

    axisX = b - a
    axisX = axisX / np.linalg.norm(axisX)
    axisY = c - a
    axisY = axisY / np.linalg.norm(axisY)
    axisZ = np.cross(axisX, axisY)
    axisZ = axisZ / np.linalg.norm(axisZ)

    return axisZ


def flipWinding(triangle):
    return [triangle[1], triangle[0], triangle[2]]


def fixMeshWinding(mesh, vertices):
    for i in range(mesh.shape[0]):
        triangle = mesh[i]
        normal = getNormal(vertices[:, triangle])
        if normal[2] > 0:
            mesh[i] = flipWinding(triangle)

    return mesh


def getShape3D(mean3DShape, blendshapes, params):
    # skalowanie
    s = params[0]
    # rotacja
    r = params[1:4]
    # przesuniecie (translacja)
    t = params[4:6]
    w = params[6:]

    # macierz rotacji z wektora rotacji, wzor Rodriguesa
    R = cv2.Rodrigues(r)[0]
    shape3D = mean3DShape + np.sum(w[:, np.newaxis, np.newaxis] * blendshapes, axis=0)

    shape3D = s * np.dot(R, shape3D)
    shape3D[:2, :] = shape3D[:2, :] + t[:, np.newaxis]

    return shape3D


def getMask(renderedImg):
    mask = np.zeros(renderedImg.shape[:2], dtype=np.uint8)


def load3DFaceModel(filename):
    faceModelFile = np.load(filename)
    mean3DShape = faceModelFile["mean3DShape"]
    mesh = faceModelFile["mesh"]
    idxs3D = faceModelFile["idxs3D"]
    idxs2D = faceModelFile["idxs2D"]
    blendshapes = faceModelFile["blendshapes"]
    mesh = fixMeshWinding(mesh, mean3DShape)

    return mean3DShape, blendshapes, mesh, idxs3D, idxs2D


def getFaceKeypoints(img, detector, predictor, maxImgSizeForDetection=640):
    imgScale = 1
    scaledImg = img
    if max(img.shape) > maxImgSizeForDetection:
        imgScale = maxImgSizeForDetection / float(max(img.shape))
        scaledImg = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)))

    # detekcja twarzy
    dets = detector(scaledImg, 1)

    if len(dets) == 0:
        return None

    shapes2D = []
    for det in dets:
        faceRectangle = rectangle(int(det.left() / imgScale), int(det.top() / imgScale), int(det.right() / imgScale),
                                  int(det.bottom() / imgScale))

        # detekcja punktow charakterystycznych twarzy
        dlibShape = predictor(img, faceRectangle)

        shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
        # transpozycja, zeby ksztalt byl 2 x n a nie n x 2, pozniej ulatwia to obliczenia
        shape2D = shape2D.T

        shapes2D.append(shape2D)

    return shapes2D


def getFaceTextureCoords(img, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor):
    projectionModel = OrthographicProjectionBlendshapes(blendshapes.shape[0])

    keypoints = getFaceKeypoints(img, detector, predictor)[0]
    modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], keypoints[:, idxs2D])
    modelParams = GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian,
                              ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], keypoints[:, idxs2D]), verbose=0)
    textureCoords = projectionModel.fun([mean3DShape, blendshapes], modelParams)

    return textureCoords


class Model:
    __metaclass__ = ABCMeta

    nParams = 0

    # zwraca wektor rezyduow przy danych parametrach modelu, wektorze wejsciowym i oczekiwanych wektorze wyjsciowym
    def residual(self, params, x, y):
        r = y - self.fun(x, params)
        r = r.flatten()

        return r

    # zwraca wartosci zwracane przez model przy danych parametrach i wektorze wejsciowym
    @abstractmethod
    def fun(self, x, params):
        pass

    # zwraca jakobian
    @abstractmethod
    def jacobian(self, params, x, y):
        pass

    # zwraca zbior przykladowych parametrow modelu
    @abstractmethod
    def getExampleParameters(self):
        pass

    # zwraca inny zbior przykladowych parametrow
    @abstractmethod
    def getInitialParameters(self):
        pass


class OrthographicProjectionBlendshapes(Model):
    nParams = 6

    def __init__(self, nBlendshapes):
        self.nBlendshapes = nBlendshapes
        self.nParams += nBlendshapes

    def fun(self, x, params):
        # skalowanie
        s = params[0]
        # rotacja
        r = params[1:4]
        # przesuniecie (translacja)
        t = params[4:6]
        w = params[6:]

        mean3DShape = x[0]
        blendshapes = x[1]

        # macierz rotacji z wektora rotacji, wzor Rodriguesa
        R = cv2.Rodrigues(r)[0]
        P = R[:2]
        shape3D = mean3DShape + np.sum(w[:, np.newaxis, np.newaxis] * blendshapes, axis=0)

        projected = s * np.dot(P, shape3D) + t[:, np.newaxis]

        return projected

    def jacobian(self, params, x, y):
        s = params[0]
        r = params[1:4]
        t = params[4:6]
        w = params[6:]

        mean3DShape = x[0]
        blendshapes = x[1]

        R = cv2.Rodrigues(r)[0]
        P = R[:2]
        shape3D = mean3DShape + np.sum(w[:, np.newaxis, np.newaxis] * blendshapes, axis=0)

        nPoints = mean3DShape.shape[1]

        # nSamples * 2 poniewaz kazdy punkt ma dwa wymiary (x i y)
        jacobian = np.zeros((nPoints * 2, self.nParams))

        jacobian[:, 0] = np.dot(P, shape3D).flatten()

        stepSize = 10e-4
        step = np.zeros(self.nParams)
        step[1] = stepSize;
        jacobian[:, 1] = ((self.fun(x, params + step) - self.fun(x, params)) / stepSize).flatten()
        step = np.zeros(self.nParams)
        step[2] = stepSize;
        jacobian[:, 2] = ((self.fun(x, params + step) - self.fun(x, params)) / stepSize).flatten()
        step = np.zeros(self.nParams)
        step[3] = stepSize;
        jacobian[:, 3] = ((self.fun(x, params + step) - self.fun(x, params)) / stepSize).flatten()

        jacobian[:nPoints, 4] = 1
        jacobian[nPoints:, 5] = 1

        startIdx = self.nParams - self.nBlendshapes
        for i in range(self.nBlendshapes):
            jacobian[:, i + startIdx] = s * np.dot(P, blendshapes[i]).flatten()

        return jacobian

    # nie uzywane
    def getExampleParameters(self):
        params = np.zeros(self.nParams)
        params[0] = 1

        return params

    def getInitialParameters(self, x, y):
        mean3DShape = x.T
        shape2D = y.T

        shape3DCentered = mean3DShape - np.mean(mean3DShape, axis=0)
        shape2DCentered = shape2D - np.mean(shape2D, axis=0)

        scale = np.linalg.norm(shape2DCentered) / np.linalg.norm(shape3DCentered[:, :2])
        t = np.mean(shape2D, axis=0) - np.mean(mean3DShape[:, :2], axis=0)

        params = np.zeros(self.nParams)
        params[0] = scale
        params[4] = t[0]
        params[5] = t[1]

        return params


def setOrtho(w, h):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, w, h, 0, -1000, 1000)
    glMatrixMode(GL_MODELVIEW)


def addTexture(img):
    textureId = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureId)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, img)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    return textureId


class FaceRenderer:
    def __init__(self, targetImg, textureImg, textureCoords, mesh):
        self.h = targetImg.shape[0]
        self.w = targetImg.shape[1]

        pygame.init()
        pygame.display.set_mode((self.w, self.h), DOUBLEBUF | OPENGL)
        setOrtho(self.w, self.h)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

        self.textureCoords = textureCoords
        self.textureCoords[0, :] /= textureImg.shape[1]
        self.textureCoords[1, :] /= textureImg.shape[0]

        self.faceTexture = addTexture(textureImg)
        self.renderTexture = addTexture(targetImg)

        self.mesh = mesh

    def drawFace(self, vertices):
        glBindTexture(GL_TEXTURE_2D, self.faceTexture)

        glBegin(GL_TRIANGLES)
        for triangle in self.mesh:
            for vertex in triangle:
                glTexCoord2fv(self.textureCoords[:, vertex])
                glVertex3fv(vertices[:, vertex])

        glEnd()

    def render(self, vertices):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.drawFace(vertices)

        data = glReadPixels(0, 0, self.w, self.h, GL_BGR, GL_UNSIGNED_BYTE)
        renderedImg = np.fromstring(data, dtype=np.uint8)
        renderedImg = renderedImg.reshape((self.h, self.w, 3))
        for i in range(renderedImg.shape[2]):
            renderedImg[:, :, i] = np.flipud(renderedImg[:, :, i])

        pygame.display.flip()
        return renderedImg


def LineSearchFun(alpha, x, d, fun, args):
    r = fun(x + alpha * d, *args)
    return np.sum(r ** 2)


def GaussNewton(x0, fun, funJack, args, maxIter=10, eps=10e-7, verbose=1):
    x = np.array(x0, dtype=np.float64)

    oldCost = -1
    for i in range(maxIter):
        r = fun(x, *args)
        cost = np.sum(r ** 2)

        if verbose > 0:
            print("Cost at iteration " + str(i) + ": " + str(cost))

        if (cost < eps or abs(cost - oldCost) < eps):
            break
        oldCost = cost

        J = funJack(x, *args)
        grad = np.dot(J.T, r)
        H = np.dot(J.T, J)
        direction = np.linalg.solve(H, grad)

        # optymalizacja dlugosci kroku
        lineSearchRes = optimize.minimize_scalar(LineSearchFun, args=(x, direction, fun, args))
        # dlugosc kroku
        alpha = lineSearchRes["x"]

        x = x + alpha * direction

    if verbose > 0:
        print("Gauss Netwon finished after " + str(i + 1) + " iterations")
        r = fun(x, *args)
        cost = np.sum(r ** 2)
        print("cost = " + str(cost))
        print("x = " + str(x))

    return x


def SteepestDescent(x0, fun, funJack, args, maxIter=10, eps=10e-7, verbose=1):
    x = np.array(x0, dtype=np.float64)

    oldCost = -1
    for i in range(maxIter):
        r = fun(x, *args)
        cost = np.sum(r ** 2)

        if verbose > 0:
            print("Cost at iteration " + str(i) + ": " + str(cost))

        # warunki stopu
        if (cost < eps or abs(cost - oldCost) < eps):
            break
        oldCost = cost

        J = funJack(x, *args)
        grad = 2 * np.dot(J.T, r)
        direction = grad

        # optymalizacja dlugosci kroku
        lineSearchRes = optimize.minimize_scalar(LineSearchFun, args=(x, direction, fun, args))
        # dlugosc kroku
        alpha = lineSearchRes["x"]

        x = x + alpha * direction

    if verbose > 0:
        print("Steepest Descent finished after " + str(i + 1) + " iterations")
        r = fun(x, *args)
        cost = np.sum(r ** 2)
        print("cost = " + str(cost))
        print("x = " + str(x))

    return x


def blendImages(src, dst, mask, featherAmount=0.2):
    # indeksy nie czarnych pikseli maski
    maskIndices = np.where(mask != 0)
    # te same indeksy tylko, ze teraz w jednej macierzy, gdzie kazdy wiersz to jeden piksel (x, y)
    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (
                1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    return composedImg


# uwaga, tutaj src to obraz, z ktorego brany bedzie kolor
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)
    # indeksy nie czarnych pikseli maski
    maskIndices = np.where(mask != 0)
    # src[maskIndices[0], maskIndices[1]] zwraca piksele w nie czarnym obszarze maski

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst


def drawPoints(img, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color)


def drawCross(img, params, center=(100, 100), scale=30.0):
    R = cv2.Rodrigues(params[1:4])[0]

    points = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    points = np.dot(points, R.T)
    points2D = points[:, :2]

    points2D = (points2D * scale + center).astype(np.int32)

    cv2.line(img, (center[0], center[1]), (points2D[0, 0], points2D[0, 1]), (255, 0, 0), 3)
    cv2.line(img, (center[0], center[1]), (points2D[1, 0], points2D[1, 1]), (0, 255, 0), 3)
    cv2.line(img, (center[0], center[1]), (points2D[2, 0], points2D[2, 1]), (0, 0, 255), 3)


def drawMesh(img, shape, mesh, color=(255, 0, 0)):
    for triangle in mesh:
        point1 = shape[triangle[0]].astype(np.int32)
        point2 = shape[triangle[1]].astype(np.int32)
        point3 = shape[triangle[2]].astype(np.int32)

        cv2.line(img, (point1[0], point1[1]), (point2[0], point2[1]), (255, 0, 0), 1)
        cv2.line(img, (point2[0], point2[1]), (point3[0], point3[1]), (255, 0, 0), 1)
        cv2.line(img, (point3[0], point3[1]), (point1[0], point1[1]), (255, 0, 0), 1)


def drawProjectedShape(img, x, projection, mesh, params, lockedTranslation=False):
    localParams = np.copy(params)

    if lockedTranslation:
        localParams[4] = 100
        localParams[5] = 200

    projectedShape = projection.fun(x, localParams)

    drawPoints(img, projectedShape.T, (0, 0, 255))
    drawMesh(img, projectedShape.T, mesh)
    drawCross(img, params)

def main():
    print("Press T to draw the keypoints and the 3D model")
    print("Press R to start recording to a video file")

    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    default_dir="liuyifei.jpg"
    image_name = filedialog.askopenfilename(title='选择含有人脸的图片', filetypes=[("png图片", "*.png"), ('jpeg图片', '*.jpeg'),('jpg图片', '*.jpg')],
                                           initialdir=(os.path.expanduser(default_dir)))
    #image_name = cv2.imread(file_path, cv2.IMREAD_COLOR)
    #image_name = "/Users/apple/PycharmProjects/001/data/wuyannzu.jpeg"
    maxImageSizeForDetection = 300

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    mean3DShape, blendshapes, mesh, idxs3D, idxs2D = load3DFaceModel("candide.npz")
    projectionModel = OrthographicProjectionBlendshapes(blendshapes.shape[0])
    modelParams = None
    lockedTranslation = False
    drawOverlay = False
    cap = cv2.VideoCapture(0)
    writer = None
    cameraImg = cap.read()[1]
    textureImg = cv2.imread(image_name)
    textureCoords = getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
    renderer = FaceRenderer(cameraImg, textureImg, textureCoords, mesh)

    while True:
        cameraImg = cap.read()[1]
        shapes2D = getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

        if shapes2D is not None:
            for shape2D in shapes2D:
                modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

                # 3D model parameter optimization
                modelParams = GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian,
                                          ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]),
                                          verbose=0)

                # rendering the model to an image
                shape3D = getShape3D(mean3DShape, blendshapes, modelParams)
                renderedImg = renderer.render(shape3D)
                mask = np.copy(renderedImg[:, :, 0])
                renderedImg = colorTransfer(cameraImg, renderedImg, mask)
                cameraImg = blendImages(renderedImg, cameraImg, mask)

                if drawOverlay:
                    drawPoints(cameraImg, shape2D.T)
                    drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams,
                                       lockedTranslation)

        if writer is not None:
            writer.write(cameraImg)

        cv2.imshow('image', cameraImg)
        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == ord("q"):
            break
        if key == ord('t'):
            drawOverlay = not drawOverlay
        if key == ord('r'):
            if writer is None:
                print("Starting video writer")
                # writer = cv2.VideoWriter("../out.avi", cv2.cv.CV_FOURCC('X', 'V', 'I', 'D'), 25, (cameraImg.shape[1], cameraImg.shape[0]))

                if writer.isOpened():
                    print("Writer succesfully opened")
                else:
                    writer = None
                    print("Writer opening failed")
            else:
                print("Stopping video writer")
                writer.release()
                writer = None


if __name__ == "__main__":
    main()
