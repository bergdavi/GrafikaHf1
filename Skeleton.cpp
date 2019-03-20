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
// Nev    : Berghammer Dávid
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

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix

	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	// layout(location = 1) in vec3 vc;

	//out vec3 color;
    out vec2 coords;

	void main() {
		// color = vc;
        coords = vp;
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

	// in vec3 color;
    in vec2 coords;
	uniform vec3 color;
    uniform int mountain;
	out vec4 outColor;		// computed color of the current pixel

	void main() {
        int y = int((coords.y + 1)*300);
        int x = int((coords.x + 1)*300) + 1000 + y*2 + y%2;
        if(mountain != 0 && y > 460) {
            y = y-460;
            float s = sin(coords.x*15)*12+12;
            if(y > s) {
                float c = max(min(1.0, 1.0-(y-s)*0.01), 0.90);
                outColor = vec4(c, c, c, 1);
            }
            else {
                outColor = vec4(color, 1);
            }
        }
        else {
            outColor = vec4(color, 1);	// computed color is the color of the primitive
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
        mat4 Mtranslate(1/wScale, 0       , 0, 0,
                        0       , 1/wScale, 0, 0,
                        0       , 0       , 0, 0,
                        0       , 0       , 0, 1);

        mat4 Mscale( 1        , 0        , 0, 0,
                     0        , 1        , 0, 0,
                     0        , 0        , 1, 0,
                     -wCenter.x,-wCenter.y, 0, 1);

        return Mtranslate * Mscale;
    }

    mat4 Minv() {
        mat4 MtranslateInv( wScale, 0     , 0, 0,
                            0     , wScale, 0, 0,
                            0     , 0     , 0, 0,
                            0     , 0     , 0, 1);

        mat4 MscaleInv( 1        , 0        , 0, 0,
                        0        , 1        , 0, 0,
                        0        , 0        , 1, 0,
                        wCenter.x, wCenter.y, 0, 1);

        return MscaleInv * MtranslateInv;
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

    void zoom(float s) {
        wScale *= s;
    }

    void pan(vec2 t) {
        wCenter = wCenter + t;
    }
};

Camera2D camera;

class KochanekBartelsCurve {
    std::vector<vec4> ctrlPoints;
    std::vector<float> ts;
    float tens = -0.1;

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
                x = fmax(fmin(x, ctrlPoints[i].x - 0.05), ctrlPoints[i - 1].x + 0.05);
                ctrlPoints.insert(ctrlPoints.begin() + i, vec4(x, y));
                ts.insert(ts.begin() + i, x);
                return i;
            }
        }
    }

    vec4 moveCtrlPopint(int idx, float x, float y) {
        vec4 cp = vec4(x, y, 0, 1);
        cp.x = fmax(fmin(cp.x, ctrlPoints[idx + 1].x - 0.05), ctrlPoints[idx - 1].x + 0.05);
        ctrlPoints[idx] = cp;
        ts[idx] = cp.x;
        return cp;
    }

    int grabCtrlPoint(float x, float y) {
        vec2 from = vec2(x, y);
        for (int i = 1; i < ctrlPoints.size()-1; i++) {
            vec4 p4 = ctrlPoints[i];
            if (length(from-vec2(p4.x, p4.y)) < 0.05) {
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
    GLuint vao, vao2;
    vec2 wTranslate = vec2(0, 0);
    float scale = 1;
    int pointCnt = 0;
    int pointCnt2 = 0;

public:
    void create() {
        glGenVertexArrays(1, &vao);
        glGenVertexArrays(1, &vao2);
   

        GLuint vbo, vbo2;

        glGenBuffers(1, &vbo);
        pointCnt = 4;
        float vertexCoords[] = { -0.03, 0, 0.03, 0, -0.03, 0.1, 0.03, 0.1};   

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, pointCnt * 2 * sizeof(float), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        glGenBuffers(1, &vbo2);
        pointCnt2 = 9;
        float vertexCoords2[] = { -0.2, 0.1, 0.2, 0.1, 0, 0.3,
                                  -0.2, 0.2, 0.2, 0.2, 0, 0.4, 
                                  -0.2, 0.3, 0.2, 0.3, 0, 0.5};

        glBindVertexArray(vao2);
        glBindBuffer(GL_ARRAY_BUFFER, vbo2);
        glBufferData(GL_ARRAY_BUFFER, pointCnt2 * 2 * sizeof(float), vertexCoords2, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
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
        mat4 Mscale(scale, 0, 0, 0,
            0, scale, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 1);

        mat4 Mtranslate(1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 0,
            wTranslate.x, wTranslate.y, 0, 1);

        return Mscale * Mtranslate;
    }

    void draw(mat4 Mat) {
        mat4 MVPTransform = Mat * M();
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

        int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
        if (colorLocation >= 0) glUniform3f(colorLocation, 0.4, 0.2, 0);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, pointCnt);

        if (colorLocation >= 0) glUniform3f(colorLocation, 0, 0.4, 0);
        glBindVertexArray(vao2);
        glDrawArrays(GL_TRIANGLES, 0, pointCnt2);
    }

    void draw() {
        mat4 Mat(1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
        draw(Mat);
    }
};

class Map {
    GLuint vao, vbo;
    GLuint vaoBg, vboBg;
    KochanekBartelsCurve *curve;
    KochanekBartelsCurve *bgCurve;
    std::vector<Tree> trees;
    std::vector<float> curveVertexCoords;
    std::vector<float> bgVertexCoords;

    void generateVertexCoord() {
        int tesselatedCount = 1000;

        curveVertexCoords.clear();
        for (int i = 0; i < tesselatedCount; i++) {
            float tNormalized = ((float)i) / (tesselatedCount - 1.0f);
            float t = curve->tStart() + (curve->tEnd() - curve->tStart())*tNormalized;
            vec4 curveCoord = curve->r(t);
            curveVertexCoords.push_back(curveCoord.x);
            curveVertexCoords.push_back(curveCoord.y);
            curveVertexCoords.push_back(curveCoord.x);
            curveVertexCoords.push_back(-2);
        }
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, curveVertexCoords.size() * sizeof(float), &curveVertexCoords[0], GL_DYNAMIC_DRAW);

        bgVertexCoords.clear();
        for (int i = 0; i < tesselatedCount; i++) {
            float tNormalized = ((float)i) / (tesselatedCount - 1.0f);
            float t = bgCurve->tStart() + (bgCurve->tEnd() - bgCurve->tStart())*tNormalized;
            vec4 curveCoord = bgCurve->r(t);
            bgVertexCoords.push_back(curveCoord.x);
            bgVertexCoords.push_back(curveCoord.y);
            bgVertexCoords.push_back(curveCoord.x);
            bgVertexCoords.push_back(-2);
        }
        glBindVertexArray(vaoBg);
        glBindBuffer(GL_ARRAY_BUFFER, vboBg);
        glBufferData(GL_ARRAY_BUFFER, bgVertexCoords.size() * sizeof(float), &bgVertexCoords[0], GL_DYNAMIC_DRAW);
    }
public:
    void create() {
        curve = new KochanekBartelsCurve(-0.2);
        bgCurve = new KochanekBartelsCurve(0.3);

        curve->setEnds(vec2(-1.1, -0.3), vec2(1.1, -0.3));
        bgCurve->setEnds(vec2(-1.1, 0.3), vec2(1.1, 0.3));

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        glGenVertexArrays(1, &vaoBg);
        glBindVertexArray(vaoBg);
        glGenBuffers(1, &vboBg);
        glBindBuffer(GL_ARRAY_BUFFER, vboBg);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        generateVertexCoord();
    }

    void addCtrlPoint(float x, float y) {
        int idx = curve -> addCtrlPoint(x, y);

        if (idx > 0) {
            y += 0.6f;
            bgCurve->addCtrlPoint(x, y);

            Tree tree;
            tree.create();
            tree.setTranslation(vec2(x, y));
            tree.setScale(0.1);
            trees.insert(trees.begin() + idx - 1, tree);
        }        
        generateVertexCoord();
    }

    void moveCtrlPoint(int idx, float x, float y) {
        vec4 moved = curve -> moveCtrlPopint(idx, x, y);
        bgCurve -> moveCtrlPopint(idx, moved.x, moved.y+0.6f);
        trees[idx-1].setTranslation(vec2(moved.x, moved.y+0.6f));

        generateVertexCoord();
    }

    int grabCtrlPoint(float x, float y) {
        return bgCurve->grabCtrlPoint(x, y);
    }

    void clear() {
        curve -> setEnds(vec2(-1.1, -0.3), vec2(1.1, -0.3));
        bgCurve -> setEnds(vec2(-1.1, 0.3), vec2(1.1, 0.3));
        trees.clear();
        generateVertexCoord();
    }

    void draw(mat4 Mat) {
        mat4 MVPTransform = Mat;
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
        int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
        if (colorLocation >= 0) glUniform3f(colorLocation, 0.7f, 0.7f, 0.7f);
        int mountainLocation = glGetUniformLocation(gpuProgram.getId(), "mountain");
        if (mountainLocation >= 0) glUniform1i(mountainLocation, 1);
        glBindVertexArray(vaoBg);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, bgVertexCoords.size() / 2);
        if (mountainLocation >= 0) glUniform1i(mountainLocation, 0);

        for (int i = 0; i < trees.size(); i++) {
            trees[i].draw(MVPTransform);
        }

        MVPTransform = Mat * camera.M();
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
        if (colorLocation >= 0) glUniform3f(colorLocation, 0.0f, 0.0f, 0.0f);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, curveVertexCoords.size()/2);
        glBindVertexArray(vao);


    }

    void draw() {
        mat4 M( 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1);
        draw(M);
    }

    vec2 r(float s) {
        vec4 v = curve -> r(s);
        return vec2(v.x, v.y);
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

        float stepRad = stepDeg * M_PI / 180.0f;
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

    void Animate(float t) {
        phi = t;
    }

    void addDistance(float dist) {
        addRotation(-dist/(r*scale));
    }

    void addRotation(float dphi) {
        phi += dphi;
    }

    void addTranslation(vec2 wT) {
        wTranslate = wTranslate + wT;
    }

    void setScale(float s) {
        scale = s;
    }

    vec2 getPedalPos(int idx) {
        return vec2(cosf(phi+idx*M_PI)*r*0.7, sinf(phi+idx*M_PI)*r*0.7);
    }

    mat4 M() {
        mat4 Mscale( scale, 0    , 0, 0,
                     0    , scale, 0, 0,
                     0    , 0    , 0, 0,
                     0    , 0    , 0, 1);

        mat4 Mrotate( cosf(phi), sinf(phi), 0, 0,
                      -sinf(phi), cosf(phi), 0, 0,
                      0        , 0        , 0, 0,
                      0        , 0        , 0, 1);

        mat4 Mtranslate( 1          , 0           , 0, 0,
                         0          , 1           , 0, 0,
                         0          , 0           , 0, 0,
                         wTranslate.x, wTranslate.y, 0, 1);

        return  Mrotate * Mscale * Mtranslate;
    }

    void draw(mat4 Mat) {
        mat4 MVPTransform = M()*Mat*camera.M();
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

        int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
        if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);

        glBindVertexArray(vao);
        glDrawArrays(GL_LINE_LOOP, 0, pointCnt);
    }

    void draw() {
        mat4 M( 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1);
        draw(M);
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
        mat4 Mtranslate(1	  ,	0	  , 0, 0,
                        0	  ,	1	  , 0, 0,
                        0	  ,	0	  , 0, 0,
                        from.x, from.y, 0, 1);

        mat4 Mskew( to.x-from.x, 0			, 0, 0,
                    0		   , to.y-from.y, 0, 0,
                    0		   , 0			, 0, 0,
                    0		   , 0			, 0, 1);

        return Mskew * Mtranslate;
    }

    void draw(mat4 Mat) {
        mat4 MVPTransform = M()*Mat*camera.M();
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

        int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
        if (colorLocation >= 0) glUniform3f(colorLocation, 0, 0.7, 0);

        glBindVertexArray(vao);
        glDrawArrays(GL_LINE_LOOP, 0, 2);
    }

    void draw() {
        mat4 M( 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1);
        draw(M);
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

    void draw() {
        mat4 M( 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1);
        draw(M);
    }

};

class Rider {
    GLuint vao;
    vec2 wTranslate = vec2(0, 0);
    float sx = 1, sy = 1;
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

    void setScale(float scale) {
        sx = scale;
        sy = scale;
    }

    mat4 M() {
        mat4 Mscale( sx, 0 , 0, 0,
                     0 , sy, 0, 0,
                     0 , 0 , 0, 0,
                     0 , 0 , 0, 1);

        mat4 Mtranslate( 1           , 0           , 0, 0,
                         0           , 1           , 0, 0,
                         0           , 0           , 0, 0,
                         wTranslate.x, wTranslate.y, 0, 1);

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

        glBindVertexArray(vao);
        glDrawArrays(GL_LINE_LOOP, 0, pointCnt);
    }

    void draw() {
        mat4 M( 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1);
        draw(M);
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

    void Animate(float t) {
        wheel.Animate(t);
        rider.setFrontFootPos(wheel.getPedalPos(0));
        rider.setBackFootPos(wheel.getPedalPos(1));
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
        mat4 Mscale( scale, 0    , 0, 0,
                     0    , scale, 0, 0,
                     0    , 0    , 0, 0,
                     0    , 0    , 0, 1);

        mat4 Mrotate( cosf(phi), sinf(phi), 0, 0,
                      -sinf(phi), cosf(phi), 0, 0,
                      0        , 0        , 0, 0,
                      0         , 0       , 0, 1);

        mat4 Mtranslate( 1           , 0           , 0, 0,
                         0           , 1           , 0, 0,
                         0           , 0           , 0, 0,
                         wTranslate.x, wTranslate.y, 0, 1);

        mat4 Mdirection(direction, 0, 0, 0,
                        0        , 1, 0, 0,
                        0        , 0, 1, 0,
                        0        , 0, 0, 1);

        return Mscale * Mrotate * Mdirection*Mtranslate;
    }

    void draw(mat4 Mat) {
        mat4 MVPTransform = Mat * M();
        mat4 t = MVPTransform;

        MVPTransform = MVPTransform * camera.M();
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
        int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
        if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
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
        grabbedPoint = map.grabCtrlPoint(x, y);
        if (grabbedPoint < 0) {
            vec4 wCoord = vec4(x, y, 0, 1) * camera.Minv();
            map.addCtrlPoint(wCoord.x, wCoord.y);
            grabbedPoint = -1;
        }   
    }

    void onMouseMoved(float x, float y) {
        if (grabbedPoint >= 0) {
            map.moveCtrlPoint(grabbedPoint, x, y-0.6);
        }
    }

    void toggleCameraFollow() {
        following = !following;
        if (following) {
            camera.setZoom(0.5);
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

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glLineWidth(2);

    game.init();

    // create program for the GPU
    gpuProgram.Create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.5, 0.5, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    game.draw();

    glutSwapBuffers();
}

// Key of ASCII code pressed
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

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

bool leftButtonDown = false;
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
    if (leftButtonDown) {
        float x = 2.0f * pX / windowWidth - 1;
        float y = 1.0f - 2.0f * pY / windowHeight;      

        game.onMouseMoved(x, y);
        glutPostRedisplay();
    }
}

// Mouse click event
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
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME);
    float sec = time / 1000.0f;
    game.timeElapsed(sec - lastSec);
    lastSec = sec;
    glutPostRedisplay();
}