//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2018. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Berghammer D�vid
// Neptun : EB2DYD
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char * const vertexSource = R"(
	#version 330				
	precision highp float;		

	uniform mat4 MVP;			

	layout(location = 0) in vec2 vp;
	layout(location = 1) in vec2 vu;
    layout(location = 2) in vec3 vc;

    out vec2 coords;
    out vec3 calcColor;

	void main() {
        coords = vu;
        calcColor = vc;
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			
	precision highp float;	

    in vec2 coords;
    in vec3 calcColor;
	uniform vec3 color;
    uniform sampler2D textureUnit;
    uniform int colorMode;
	out vec4 outColor;		

	void main() {       
        if(colorMode == 0) {
            outColor = texture(textureUnit, coords);           
        }
        else if(colorMode == 1){
            outColor = vec4(color, 1);	
        }
        else if(colorMode == 2) {
            outColor = vec4(calcColor, 1);
        }
	}
)";

GPUProgram gpuProgram;

class Camera2D {
    vec2 wCenter;
    float wScale;
public:
    Camera2D() : wCenter(0, 0), wScale(1) {}

    mat4 M() {
        mat4 Mscale = ScaleMatrix(vec3(1/wScale, 1/wScale));
        mat4 Mtranslate = TranslateMatrix(vec3(-wCenter));

        return Mscale * Mtranslate;
    }

    mat4 Minv() {
        mat4 MscaleInv = ScaleMatrix(vec3(wScale, wScale));
        mat4 MtranslateInv = TranslateMatrix(wCenter);

        return MtranslateInv * MscaleInv;
    }

    float getZoom() {
        return wScale;
    }

    void setZoom(float s) {
        wScale = s;
    }

    void setPan(vec2 t) {
        wCenter = t * (1.0f/wScale);
    }
};

Camera2D camera;

class KochanekBartelsCurve {
    std::vector<vec4> ctrlPoints;
    std::vector<float> ts;
    float tens;

    std::vector<vec4> HermiteConstants(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t) {
        vec4 a0 = p0;
        vec4 a1 = v0;
        vec4 a2 = (p1 - p0) * 3 / pow((t1 - t0), 2) - (v1 + v0 * 2) / (t1 - t0);
        vec4 a3 = (p0 - p1) * 2 / pow((t1 - t0), 3) + (v1 + v0) / pow((t1 - t0), 2);

        std::vector<vec4> results;
        results.push_back(a0); results.push_back(a1); results.push_back(a2); results.push_back(a3);
        return results;
    }

    vec4 Hermite(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t) {
        std::vector<vec4> a = HermiteConstants(p0, v0, t0, p1, v1, t1, t);
        vec4 rt = a[3] * pow(t - t0, 3) + a[2] * pow(t - t0, 2) + a[1] * (t - t0) + a[0];
        return rt;
    }

    vec4 dHermite(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t) {
        std::vector<vec4> a = HermiteConstants(p0, v0, t0, p1, v1, t1, t);
        vec4 rt = a[3] * pow(t - t0, 2) * 3 + a[2] * (t - t0) * 2 + a[1];
        return rt;
    }
public:
    KochanekBartelsCurve(float tens) : tens(tens) {}

    float tStart() { return ts[0]; }
    float tEnd() { return ts[ctrlPoints.size() - 1]; }

    void setEnds(vec2 start, vec2 end) {
        ctrlPoints.clear();
        ts.clear();
        ctrlPoints.push_back(vec4(start.x, start.y, 0, 1));
        ctrlPoints.push_back(vec4(end.x, end.y, 0, 1));
        ts.push_back(start.x);
        ts.push_back(end.x);
    }

    int addCtrlPoint(float &x, float &y) {
        for (int i = 0; i < ctrlPoints.size(); i++) {
            vec4 p = ctrlPoints[i];
            if (p.x > x) {
                if (ctrlPoints[i].x - ctrlPoints[i-1].x < 0.1) {
                    return -1;
                }
                x = fmaxf(fminf(x, ctrlPoints[i].x - 0.05f), ctrlPoints[i - 1].x + 0.05f);
                ctrlPoints.insert(ctrlPoints.begin() + i, vec4(x, y));
                ts.insert(ts.begin() + i, x);
                return i;
            }
        }
    }

    vec4 moveCtrlPoint(int idx, float x, float y) {
        vec4 cp = vec4(x, y, 0, 1);
        cp.x = fmaxf(fminf(cp.x, ctrlPoints[idx + 1].x - 0.05f), ctrlPoints[idx - 1].x + 0.05f);
        ctrlPoints[idx] = cp;
        ts[idx] = cp.x;
        return cp;
    }

    int grabCtrlPoint(float x, float y, float r) {
        vec2 from = vec2(x, y);
        for (int i = 1; i < ctrlPoints.size()-1; i++) {
            vec4 p4 = ctrlPoints[i];
            if (length(from-vec2(p4.x, p4.y)) < r) {
                return i;
            }
        }
        return -1;
    }

    vec4 r(float t, bool derivative = false) {
        t = fmax(t, tStart());
        t = fmin(t, tEnd());
        for (int i = 0; i < ctrlPoints.size() - 1; i++) {
            if (ts[i] <= t && t <= ts[i + 1]) {
                vec4 v0 = vec4(1, 0, 0, 1);
                vec4 v1 = vec4(1, 0, 0, 1);
                vec4 p0 = ctrlPoints[i];
                vec4 p1 = ctrlPoints[i + 1];
                float t0 = ts[i];
                float t1 = ts[i + 1];
                if (i > 0) {
                    vec4 p_1 = ctrlPoints[i - 1];
                    float t_1 = ts[i - 1];
                    v0 = ((p1 - p0) / (t1 - t0) + (p0 - p_1) / (t0 - t_1)) *(1 - tens);
                }
                if (i < ctrlPoints.size() - 2) {
                    vec4 p2 = ctrlPoints[i + 2];
                    float t2 = ts[i + 2];
                    v1 = ((p2 - p1) / (t2 - t1) + (p1 - p0) / (t1 - t0)) *(1 - tens);
                }
                if (derivative) {
                    return dHermite(p0, v0, t0, p1, v1, t1, t);
                }
                return Hermite(p0, v0, t0, p1, v1, t1, t);
            }
        }
    }
};

class Tree {
    GLuint vaoTrunk, vaoLeaves;
    vec2 wTranslate = vec2(0, 0);
    float scale = 1;
    int pointCntTrunk = 0;
    int pointCntLeaves = 0;

public:
    void create() {
        glGenVertexArrays(1, &vaoTrunk);
        glGenVertexArrays(1, &vaoLeaves);
        GLuint vboTrunk, vboLeaves, vboColorLeaves;

        glGenBuffers(1, &vboTrunk);
        pointCntTrunk = 4;
        float vertexCoordsTrunk[] = { -0.02, 0, 0.02, 0, -0.02, 0.1, 0.02, 0.1};

        glBindVertexArray(vaoTrunk);
        glBindBuffer(GL_ARRAY_BUFFER, vboTrunk);
        glBufferData(GL_ARRAY_BUFFER, pointCntTrunk * 2 * sizeof(float), vertexCoordsTrunk, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        glGenBuffers(1, &vboLeaves);
        glGenBuffers(1, &vboColorLeaves);
        pointCntLeaves = 9;
        float vertexCoordsLeaves[] = { -0.15, 0.1, 0.15, 0.1, 0, 0.3,
                                       -0.15, 0.2, 0.15, 0.2, 0, 0.4,
                                       -0.15, 0.3, 0.15, 0.3, 0, 0.5};
        float *vertexColorsLeaves = new float[pointCntLeaves * 3];
        for(int i = 0; i < pointCntLeaves; i++) {
            vertexColorsLeaves[i*3] = 0;
            vertexColorsLeaves[i*3+2] = 0;
            if(i%3 == 2) {
                vertexColorsLeaves[i*3+1] = 0.6;
            }
            else {
                vertexColorsLeaves[i*3+1] = 0.4;
            }
        }

        glBindVertexArray(vaoLeaves);
        glBindBuffer(GL_ARRAY_BUFFER, vboLeaves);
        glBufferData(GL_ARRAY_BUFFER, pointCntLeaves * 2 * sizeof(float), vertexCoordsLeaves, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        glBindBuffer(GL_ARRAY_BUFFER, vboColorLeaves);
        glBufferData(GL_ARRAY_BUFFER, pointCntLeaves * 3 * sizeof(float), vertexColorsLeaves, GL_STATIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);

        delete vertexColorsLeaves;
    }

    void addTranslation(vec2 wT) {
        wTranslate = wTranslate + wT;
    }

    void setTranslation(vec2 wT) {
        addTranslation(wT - wTranslate);
    }

    void setScale(float s) {
        scale = s;
    }

    mat4 M() {
        mat4 Mscale = ScaleMatrix(vec3(scale, scale));
        mat4 Mtranslate = TranslateMatrix(wTranslate);

        return Mscale * Mtranslate;
    }

    void draw(mat4 Mat) {
        mat4 MVPTransform = Mat * M() * camera.M();
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

        int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
        if (colorLocation >= 0) glUniform3f(colorLocation, 0.4, 0.2, 0);
        int colorModeLocation = glGetUniformLocation(gpuProgram.getId(), "colorMode");
        if (colorModeLocation >= 0) glUniform1i(colorModeLocation, 1);

        glBindVertexArray(vaoTrunk);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, pointCntTrunk);

        if (colorModeLocation >= 0) glUniform1i(colorModeLocation, 2);
        glBindVertexArray(vaoLeaves);
        glDrawArrays(GL_TRIANGLES, 0, pointCntLeaves);
    }
};

class Background {
    unsigned int vao, vbo[2];
    Texture * pTexture;
public:   
    void create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(2, vbo);

        float vertices[] = {-1, -1, -1, 1, 1, 1, 1, -1};
        float uvs[] = {0, 0, 0, 1, 1, 1, 1, 0};

        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        int width = 600, height = 600;
        std::vector<vec4> image(width * height);
        vec4 skyColor = vec4(0.2, 0.4, 0.7, 1);
        vec4 mountainColor = vec4(0.5f, 0.5f, 0.5f, 1);

        KochanekBartelsCurve *curve = new KochanekBartelsCurve(0.7);
        curve -> setEnds(vec2(0, 400), vec2(600, 350));
        float cpX = 150;
        float cpY = 570;
        curve ->addCtrlPoint(cpX, cpY);
        cpX = 350;
        cpY = 250;
        curve -> addCtrlPoint(cpX, cpY);
        cpX = 450;
        cpY = 500;
        curve -> addCtrlPoint(cpX, cpY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (y < curve -> r(x).y) {
                    float s = sinf(x/20.0f)*10 + 420;
                    if(y > s) {
                        float c = fmaxf(fminf(1.0f, 1.0f-(y-s)*0.01f), 0.85f);
                        image[y*width + x] = vec4(c,c,c,1);
                    }
                    else {
                        image[y*width + x] = mountainColor;
                    }
                }
                else {
                    image[y*width + x] = skyColor;
                }
            }
        }

        pTexture = new Texture(width, height, image);
        delete curve;
    }



    void draw() {
        glBindVertexArray(vao);
        mat4 MVPTransform = mat4(1, 0, 0, 0, 
                                 0, 1, 0, 0, 
                                 0, 0, 1, 0,
                                 0, 0, 0, 1);
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
        int colorModeLocation = glGetUniformLocation(gpuProgram.getId(), "colorMode");
        if (colorModeLocation >= 0) glUniform1i(colorModeLocation, 0);
        pTexture->SetUniform(gpuProgram.getId(), "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

class Map {
    GLuint vao, vbo;
    KochanekBartelsCurve *curve;
    Background background;
    std::vector<Tree> trees;
    std::vector<float> curveVertexCoords;

    void generateVertexCoord() {
        int tesselatedCount = 1000;

        curveVertexCoords.clear();
        for (int i = 0; i < tesselatedCount; i++) {
            float tNormalized = ((float)i) / (tesselatedCount - 1.0f);
            float t=  curve->tStart() + (curve->tEnd() - curve->tStart())*tNormalized;
            vec4 curveCoord = curve->r(t);
            curveVertexCoords.push_back(curveCoord.x);            
            curveVertexCoords.push_back(curveCoord.y);
            curveVertexCoords.push_back(curveCoord.x);            
            curveVertexCoords.push_back(-2);
        }
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, curveVertexCoords.size() * sizeof(float), &curveVertexCoords[0], GL_DYNAMIC_DRAW);
    }
public:
    void create() {
        background.create();
        curve = new KochanekBartelsCurve(-0.2);

        curve->setEnds(vec2(-1.1, -0.8), vec2(1.1, -0.8));

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        generateVertexCoord();
    }

    void addCtrlPoint(float x, float y) {
        int idx = curve -> addCtrlPoint(x, y);

        if (idx > 0) {
            Tree tree;
            tree.create();
            tree.setTranslation(vec2(x, y-0.05));
            tree.setScale(1);
            trees.insert(trees.begin() + idx - 1, tree);
        }        
        generateVertexCoord();
    }

    void moveCtrlPoint(int idx, float x, float y) {
        vec4 moved = curve->moveCtrlPoint(idx, x, y);
        trees[idx-1].setTranslation(vec2(moved.x, moved.y-0.05));

        generateVertexCoord();
    }

    int grabCtrlPoint(float x, float y) {
        return curve->grabCtrlPoint(x, y-0.05, 0.1);
    }

    void clear() {
        curve -> setEnds(vec2(-1.1, -0.8), vec2(1.1, -0.8));
        trees.clear();
        generateVertexCoord();
    }

    void draw(mat4 Mat) {
        background.draw();
        mat4 MVPTransform = Mat;
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

        for (int i = 0; i < trees.size(); i++) {
            trees[i].draw(Mat);
        }

        MVPTransform = Mat * camera.M();
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
        int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
        if (colorLocation >= 0) glUniform3f(colorLocation, 0.0f, 0.0f, 0.0f);
        int colorModeLocation = glGetUniformLocation(gpuProgram.getId(), "colorMode");
        if (colorModeLocation >= 0) glUniform1i(colorModeLocation, 1);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, curveVertexCoords.size()/2);
    }

    void draw() {
        mat4 M( 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1);
        draw(M);
    }

    vec2 dr(float s) {
        vec4 v = curve -> r(s, true);
        return vec2(v.x, v.y);
    }

    vec2 getCoords(float s, float offset) {
        vec4 c = curve -> r(s);
        vec4 v = curve -> r(s, true);       
        return vec2(c.x, c.y) + normalize(vec2(-v.y, v.x) )* offset;
    }
};

class Wheel {
    GLuint vao;
    vec2 wTranslate = vec2(0,0);
    float phi = 0;
    float scale = 1;
    int pointCnt = 0;
    float r = 0.5;
public:
    void create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        GLuint vbo;
        glGenBuffers(1, &vbo);

        int stepDeg = 5;
        int spokeCnt = 6;

        float stepRad = (float)(stepDeg * M_PI / 180.0f);
        int circlePointCnt = 360 / stepDeg;
        int spokeFreq = circlePointCnt / spokeCnt;
        pointCnt = circlePointCnt + spokeCnt*2;

        float* vertexCoords = new float[pointCnt*2];
        for (int i = 0, j = 0; i < pointCnt; i++, j++) {
            vertexCoords[2 * i] = r * cosf(j*stepRad);
            vertexCoords[2 * i+1] = r * sinf(j*stepRad);

            if (j % spokeFreq == 0) {
                i++;
                vertexCoords[2 * i] = 0;
                vertexCoords[2 * i + 1] = 0;

                i++;
                vertexCoords[2 * i] = r * cosf(j*stepRad);
                vertexCoords[2 * i + 1] = r * sinf(j*stepRad);
            }
        }
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, pointCnt*2*sizeof(float), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    float getR() {
        return r;
    }


    void addDistance(float dist) {
        addRotation(-dist/(r*scale));
    }

    void addRotation(float dphi) {
        phi += dphi;
    }

    vec2 getPedalPos(int idx) {
        return vec2(cosf(phi+idx*M_PI)*r*0.7, sinf(phi+idx*M_PI)*r*0.7);
    }

    mat4 M() {
        mat4 Mscale = ScaleMatrix(vec3(scale, scale));
        mat4 Mrotate = RotationMatrix(phi, vec3(0,0,1));
        mat4 Mtranslate = TranslateMatrix(wTranslate);

        return  Mrotate * Mscale * Mtranslate;
    }

    void draw(mat4 Mat) {
        mat4 MVPTransform = M()*Mat*camera.M();
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

        int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
        if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
        int colorModeLocation = glGetUniformLocation(gpuProgram.getId(), "colorMode");
        if (colorModeLocation >= 0) glUniform1i(colorModeLocation, 1);
        glBindVertexArray(vao);
        glDrawArrays(GL_LINE_LOOP, 0, pointCnt);
    }
};

class LegPart {
    GLuint vao;
    vec2 from = vec2(0, 0);
    vec2 to = vec2(0, 0);

public:
    void create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        GLuint vbo;
        glGenBuffers(1, &vbo);

        float vertexCoords[] = { 0, 0, 1, 1};

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, 2 * 2 * sizeof(float), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void setFrom(vec2 v) {
        from = v;
    }

    void setTo(vec2 v) {
        to = v;
    }

    mat4 M() {
        mat4 Mtranslate = TranslateMatrix(from);
        mat4 Mskew = ScaleMatrix(to-from);

        return Mskew * Mtranslate;
    }

    void draw(mat4 Mat) {
        mat4 MVPTransform = M()*Mat*camera.M();
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

        int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
        if (colorLocation >= 0) glUniform3f(colorLocation, 0, 0.7, 0);
        int colorModeLocation = glGetUniformLocation(gpuProgram.getId(), "colorMode");
        if (colorModeLocation >= 0) glUniform1i(colorModeLocation, 1);
        glBindVertexArray(vao);
        glDrawArrays(GL_LINE_LOOP, 0, 2);
    }
};

class Leg {
    LegPart lower;
    LegPart upper;
    vec2 top;
    vec2 center;
    vec2 bottom;
    float lowerLength;
    float upperLength;

public:
    void create() {
        top = vec2(0, 0);
        center = vec2(0, -0.7);
        bottom = vec2(0, -1.4);
        lowerLength = length(center - top);
        upperLength = length(center - bottom);



        upper.create();
        lower.create();
        upper.setFrom(top);
        upper.setTo(center);
        lower.setFrom(center);
        lower.setTo(bottom);
    }

    void updateCenter() {
        float r0 = lowerLength;
        float r1 = upperLength;
        float d = length(bottom-top);
        float a = (r0*r0 - r1 * r1 + d * d) / (2 * d);
        float c = sqrtf(r0*r0 - a * a);
        vec2 p = normalize(bottom - top)*a;
        vec2 q = vec2(-p.y, p.x);
        center = top + p + normalize(q)*c;
        upper.setTo(center);
        lower.setFrom(center);
    }

    void setBottom(vec2 v) {
        bottom = v;
        lower.setTo(bottom);
        updateCenter();
    }

    void draw(mat4 Mat) {
        upper.draw(Mat);
        lower.draw(Mat);
    }
};

class Rider {
    GLuint vao;
    vec2 wTranslate = vec2(0, 0);
    float scale = 1;
    int pointCnt = 0;

    Leg frontLeg;
    Leg backLeg;
public:
    void create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        GLuint vbo;
        glGenBuffers(1, &vbo);

        int stepDeg = 5;
        float r = 0.2;

        float stepRad = stepDeg * M_PI / 180.0f;
        int circlePointCnt = 360 / stepDeg +1 ;
        pointCnt = circlePointCnt + 1;

        float* vertexCoords = new float[pointCnt * 2];
        for (int i = 0; i < circlePointCnt; i++) {
            vertexCoords[2 * i] = r * cosf(i*stepRad - M_PI_2);
            vertexCoords[2 * i + 1] = r * sinf(i*stepRad - M_PI_2)+1.25;
        }

        vertexCoords[2 * circlePointCnt] = 0;
        vertexCoords[2 * circlePointCnt + 1] = 0;

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, pointCnt * 2 * sizeof(float), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        frontLeg.create();
        backLeg.create();
    }

    void setFrontFootPos(vec2 pos) {
        frontLeg.setBottom(pos - wTranslate);
    }

    void setBackFootPos(vec2 pos) {
        backLeg.setBottom(pos - wTranslate);
    }

    void addTranslation(vec2 wT) {
        wTranslate = wTranslate + wT;
    }

    mat4 M() {
        mat4 Mscale = ScaleMatrix(vec3(scale, scale));
        mat4 Mtranslate = TranslateMatrix(wTranslate);

        return Mscale * Mtranslate;
    }

    void draw(mat4 Mat) {
        mat4 MVPTransform = M()*Mat;

        backLeg.draw(MVPTransform);
        frontLeg.draw(MVPTransform);

        MVPTransform = MVPTransform * camera.M();

        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

        int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
        if (colorLocation >= 0) glUniform3f(colorLocation, 0, 0.7, 0);
        int colorModeLocation = glGetUniformLocation(gpuProgram.getId(), "colorMode");
        if (colorModeLocation >= 0) glUniform1i(colorModeLocation, 1);
        glBindVertexArray(vao);
        glDrawArrays(GL_LINE_LOOP, 0, pointCnt);
    }
};

class Unicycle {
    GLuint vao;
    vec2 wTranslate = vec2(0, 0);
    float phi = 0;
    float scale = 1;
    int pointCnt = 0;
    Wheel wheel;
    Rider rider;
    int direction = 1;
    float m = 50;
    float F = 700;

public:
    void create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        GLuint vbo;
        glGenBuffers(1, &vbo);

        pointCnt = 4;

        float vertexCoords[] = {0, 0, 0, 1, -0.3, 1, 0.3, 1};

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, pointCnt * 2 * sizeof(float), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        wheel.create();
        rider.create();
        rider.addTranslation(vec2(0,1.02));
    }

    int getDirection() { return direction; }
    float getMass() { return m; }
    float getForce() { return F; }

    float getOffset() {
        return wheel.getR()*scale;
    }

    float getCoM() {
        return 1.3*scale;
    }

    void addTranslation(vec2 wT) {
        wTranslate = wTranslate + wT;

        wheel.addDistance(length(wT)/scale);
        rider.setFrontFootPos(wheel.getPedalPos(0));
        rider.setBackFootPos(wheel.getPedalPos(1));

        if (wTranslate.x > 1) {
            direction = -1;
        }
        if (wTranslate.x < -1) {
            direction = 1;
        }
    }

    void setRotation(float p) {
        phi = p;
    }


    void setTranslation(vec2 wT) {
        addTranslation(wT-wTranslate);
    }

    void setScale(float s) {
        scale = s;
    }

    mat4 M() {
        mat4 Mscale = ScaleMatrix(vec3(scale, scale));
        mat4 Mrotate = RotationMatrix(phi, vec3(0,0,1));
        mat4 Mtranslate = TranslateMatrix(wTranslate);
        mat4 Mdirection = ScaleMatrix(vec3(direction, 1));

        return Mscale * Mrotate * Mdirection*Mtranslate;
    }

    void draw(mat4 Mat) {
        mat4 MVPTransform = Mat * M();
        mat4 t = MVPTransform;

        MVPTransform = MVPTransform * camera.M();
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
        int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
        if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
        int colorModeLocation = glGetUniformLocation(gpuProgram.getId(), "colorMode");
        if (colorModeLocation >= 0) glUniform1i(colorModeLocation, 1);
        glBindVertexArray(vao);
        glDrawArrays(GL_LINE_STRIP, 0, pointCnt);

        wheel.draw(t);
        rider.draw(t);
    }

    void draw() {
        mat4 Mat(1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 0,
                 0, 0, 0, 1);
        draw(Mat);
    }
};

class Game {
    Map map;
    float s = 0;
    float g = 9.81;
    float rho = 1400;
    bool following = false;
    int grabbedPoint = -1;
public:
    Unicycle unicycle;
    void init() {
        unicycle.create();
        unicycle.setScale(0.09);
        map.create();
    }

    void draw() {
        map.draw();
        unicycle.draw();
    }

    void onMouseDown(float x, float y) {
        vec4 wCoord = vec4(x, y, 0, 1) * camera.Minv();
        if (!following) {
            grabbedPoint = map.grabCtrlPoint(wCoord.x, wCoord.y);
        }        
        if (grabbedPoint < 0) {           
            map.addCtrlPoint(wCoord.x, wCoord.y);
            grabbedPoint = -1;
        }   
    }

    void onMouseMoved(float x, float y) {
        if (grabbedPoint >= 0) {
            vec4 wCoord = vec4(x, y, 0, 1) * camera.Minv();
            map.moveCtrlPoint(grabbedPoint, wCoord.x, wCoord.y);
        }
    }

    void toggleCameraFollow() {
        following = !following;
        if (following) {
            camera.setZoom(0.5);
            grabbedPoint = -1;
        }
        else {
            camera.setZoom(1);
            camera.setPan(vec2(0, 0));
        }
    }

    void clearMap() {
        map.clear();
    }

    void timeElapsed(float timeChanged) {
        const float dT = 0.01;
        for (float t = 0; t < timeChanged; t += dT) {
            float dt = fmin(dT, timeChanged - t);
            float speed = (unicycle.getForce() - normalize(map.dr(s)).y*unicycle.getMass()*g*unicycle.getDirection())/rho*unicycle.getDirection();
            float ds = speed*dt;
            vec2 dr = map.dr(s);
            s += ds/ length(dr);
            vec2 pos = map.getCoords(s, unicycle.getOffset());
            float ang = -asinf(unicycle.getOffset()/unicycle.getCoM()*dr.y/length(dr))*unicycle.getDirection();
            if (following) {
                camera.setPan(vec2(fmin(fmax(pos.x, -camera.getZoom()), camera.getZoom()), pos.y));
            }
            unicycle.setTranslation(pos);
            unicycle.setRotation(ang);
        }
    }
};

Game game;

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glLineWidth(2);
    printf("[Left mouse button]: Place control point\n");
    printf("[Left mouse button on tree trunk]: Move control point, (only when follow camera is off)\n");
    printf("[Space]: Toggle follow camera\n");
    printf("[Delete]: Remove all control points\n");
    game.init();
    gpuProgram.Create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
    glClearColor(0.5, 0.5, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    game.draw();

    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
    switch (key) {
        case ' ':
            game.toggleCameraFollow();
            break;
        case 127:
            game.clearMap();
            break;
        default:
            break;
    }
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

bool leftButtonDown = false;
void onMouseMotion(int pX, int pY) {
    if (leftButtonDown) {
        float x = 2.0f * pX / windowWidth - 1;
        float y = 1.0f - 2.0f * pY / windowHeight;      

        game.onMouseMoved(x, y);
        glutPostRedisplay();
    }
}

void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        leftButtonDown = true;
        float x = 2.0f * pX / windowWidth - 1;
        float y = 1.0f - 2.0f * pY / windowHeight;

        game.onMouseDown(x, y);
        glutPostRedisplay();
    }
    if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
        leftButtonDown = false;
        glutPostRedisplay();
    }
}

static float lastSec = 0;
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME);
    float sec = time / 1000.0f;
    game.timeElapsed(sec - lastSec);
    lastSec = sec;
    glutPostRedisplay();
}