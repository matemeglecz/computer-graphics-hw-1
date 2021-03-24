//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
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
// Nev    : Meglécz Máté
// Neptun : A7RBKU
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
	layout(location = 0) in vec3 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec2 vertexUV;

	out vec2 texCoord;

	void main() {
		texCoord=vertexUV;
		gl_Position = vec4(vp.x/vp.z, vp.y/vp.z, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform sampler2D textureUnit;
	uniform vec3 color;		// uniform variable, the color of the primitive
	uniform int isPoint;
	
	in vec2 texCoord;
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		if(isPoint==1){
			outColor = texture(textureUnit, texCoord);
		} else {
			outColor = vec4(color, 1);	// computed color is the color of the primitive
		}
	}
)";

#define FLT_MIN		1.175494351e-38F        // min normalized positive value

static const int VERTICES_NUM = 50;
static const double FULLNESS = 0.05;

static const int LINES_NUM = round(VERTICES_NUM * (VERTICES_NUM - 1) / 2 * FULLNESS);

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU
unsigned int vbo[2];		// vertex buffer object


float distanceHyper(vec3 p, vec3 q) {
	return acosh((-1) * (p.x * q.x + p.y * q.y - p.z * q.z));
}

vec3 vSectionHyper(vec3 p, vec3 r, float dist) {
	return (r - p * cosh(dist)) / sinh(dist);
}

vec3 rSectionHyper(vec3 p, vec3 v, float dist) {
	return (p * cosh(dist) + v * sinh(dist));
}

vec3 mirrorHyper(vec3 p, vec3 m) {
	float dist = distanceHyper(p, m);
	vec3 v = vSectionHyper(p, m, dist);
	return rSectionHyper(p, v, 2 * dist);
}

class Line {
public:
	vec3* vertex1;
	vec3* vertex2;
	bool used;

	Line(vec3* v1, vec3* v2, bool u= false) {
		vertex1 = v1;
		vertex2 = v2;
		used = u;
	}
};


class Graph {
private:
	std::vector<vec2> uvs;
	float dAngle = 0.05;
	
public:
	std::vector<vec3> vertices;
	std::vector<vec3> verticesV;
	std::vector<Line> lines;
	std::vector<Texture*> textures;

	Graph() {
		for (int i = 0; i < VERTICES_NUM; i++) {
			verticesV.push_back(vec3(0, 0, 0));
		}

		vec2 firstUVS;
		for (double i = 0; i < M_PI*2; i += dAngle) {
			float x = 0.5 + 0.5 * cos(i);
			float y = 0.5 + 0.5 * sin(i);
			uvs.push_back(vec2(x,y));
			if (i == 0) firstUVS = vec2(x, y);
		}
		uvs.push_back(firstUVS);
	}

	void drawPoints() {
		glBindVertexArray(vao);

		float circleR = 0.05;

		for (int i = 0; i < VERTICES_NUM; i++) {
			std::vector<vec3> circlePoints;
			vec2 center = vec2((vertices[i].x / vertices[i].z), (vertices[i].y / vertices[i].z));
			vec3 firstPoint;
			for (double j = 0; j < 2 * M_PI; j += dAngle) {
				vec3 vectorR=vec3(0,0,1);
				vectorR.x = circleR*cos(j);
				vectorR.y = circleR*sin(j);
				vectorR = vectorR / (float)sqrt(1 - vectorR.x * vectorR.x - vectorR.y * vectorR.y);
				vec3 p = vec3(0, 0, 1);

				float dist = distanceHyper(p, vectorR);
				vec3 v = vSectionHyper(p, vectorR, dist);

				//m1 point
				vec3 m1 = p * cosh(dist / 4) + v * sinh(dist / 4);
				//m2 point
				vec3 m2 = p * cosh(dist * 3 / 4) + v * sinh(dist * 3 / 4);

				vec3 circlePoint=vertices[i];
				circlePoint = mirrorHyper(circlePoint, m1);
				circlePoint = mirrorHyper(circlePoint, m2);
				circlePoints.push_back(circlePoint);
				if (j == 0) {
					firstPoint = circlePoint;
				}
			}
			circlePoints.push_back(firstPoint);

			glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
			glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
				circlePoints.size() * sizeof(vec3),  // # bytes
				&circlePoints[0],	      	// address
				GL_STATIC_DRAW);	// we do not change later

			glEnableVertexAttribArray(0);  // AttribArray 0
			glVertexAttribPointer(0,       // vbo -> AttribArray 0
				3, GL_FLOAT, GL_FALSE, // three floats/attrib, not fixed-point
				0, NULL); 		     // stride, offset: tightly packed


			glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
			glBufferData(GL_ARRAY_BUFFER, uvs.size()*sizeof(vec2), &uvs[0], GL_STATIC_DRAW);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

			gpuProgram.setUniform(*textures[i], "textureUnit");

			glBindVertexArray(vao);  // Draw call*/
			glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, circlePoints.size() /*# Elements*/);
		}
	}

	void drawLines() {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

		std::vector<vec3> usedLines;
		for (size_t i = 0; i < lines.size(); i++)
		{
			if (lines[i].used) {
				usedLines.push_back(*lines[i].vertex1);
				usedLines.push_back(*lines[i].vertex2);
			}
		}

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			usedLines.size() * sizeof(vec3),  // # bytes
			&usedLines[0],	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			3, GL_FLOAT, GL_FALSE, // three floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed


		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_LINES, 0 /*startIdx*/, LINES_NUM * 2 /*# Elements*/);
	}

	void drawGraph() {
		int color = glGetUniformLocation(gpuProgram.getId(), "color");
		
		gpuProgram.setUniform(0, "isPoint"); // 1 float
		glUniform3f(color, 1.0f, 1.0f, 0.0f); //lines are yellow
		drawLines();

		gpuProgram.setUniform(1, "isPoint");
		drawPoints();
	}

	
};

int pressedButton;
Graph graph;
vec2 vectorStart;

Texture* TextureGen(vec3 point) {
	int width = 20, height = 20;			
	std::vector<vec4> image(width * height);
	float r, g, b;
	for (int y = 0; y < height; y++) {
		if (y % 5 == 0) {
			r = (float)rand() / RAND_MAX;
			g = (float)rand() / RAND_MAX;
			b = (float)rand() / RAND_MAX;
		}
		for (int x = 0; x < width; x++) {
			image[y * width + x] = vec4(r, g, b, 1);
		}
	}
	return  new Texture(width, height, image);
}

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glGenBuffers(2, vbo);	
	
	graph = Graph();

	for (int i = 0; i < VERTICES_NUM; i++) {
		float x = (((float)rand() * 2 / (float)(RAND_MAX)) - 1.0f);
		float y = (((float)rand() * 2 / (float)(RAND_MAX)) - 1.0f);

		graph.vertices.push_back(vec3(x, y, (float)sqrt(1 + x*x + y * y)));
		graph.textures.push_back(TextureGen(graph.vertices[i]));
	}

	//lines generate
	for (int i = 0; i < VERTICES_NUM; i++) {
		for (int j = 0; j < i; j++) {
			graph.lines.push_back(Line(&graph.vertices[i], &graph.vertices[j]));
		}
	}

	for (int i = 0; i < LINES_NUM; i++) {
		bool success = false;
		while (!success) {
			int randidx = rand() % (VERTICES_NUM * (VERTICES_NUM - 1) / 2);
			if (!graph.lines[randidx].used) {
				graph.lines[randidx].used = true;
				success = true;
			}
		}
	}

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glPointSize(8.0f);
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	int location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	graph.drawGraph(); // gráf rajzolása

	glutSwapBuffers(); // exchange buffers for double buffering
}

void KMeans() { 
	for (size_t i = 0; i < VERTICES_NUM; i++){
		vec3 center= vec3(0, 0, 0);

		for (size_t j = 0; j < VERTICES_NUM; j++){
			if (i != j) {
				vec3 nextV = graph.vertices[j];
				bool neighbour = false;
				for (size_t z = 0; z < graph.lines.size(); z++){
					if (((graph.lines[z].vertex1 == &graph.vertices[i] && graph.lines[z].vertex2 == &graph.vertices[j])
						|| (graph.lines[z].vertex1 == &graph.vertices[j] && graph.lines[z].vertex2 == &graph.vertices[i])) && graph.lines[z].used) {
						neighbour = true;
						break;
					}
				}
				if (neighbour) {
					center = center + (nextV/nextV.z);
				}
				else center = center - (nextV / nextV.z);
			}
		}
		center = center / (VERTICES_NUM - 1);
		center.z = 1;
	
		center = center / sqrt(1 - center.x * center.x - center.y * center.y);
		
		if (abs(center.x) >= FLT_MIN && abs(center.y) >= FLT_MIN && abs(center.z) >= FLT_MIN) {
			graph.vertices[i] = center;
		}
	}
}


void graphMove(float cx, float cy) {
	if (pressedButton != GLUT_RIGHT_BUTTON) return;
	float x = (cx - vectorStart.x);
	float y = (cy - vectorStart.y);
	if ((cx != vectorStart.x || cy != vectorStart.y) && abs(x) > FLT_MIN && abs(y) > FLT_MIN) {
		vectorStart.x = cx;
		vectorStart.y = cy;

		float w = sqrt(1 - x * x - y * y);
		vec3 moveVec = vec3(x / w, y / w, 1 / w);

		if (abs(x) < FLT_MIN || abs(y) < FLT_MIN) {
			printf("problem\n");
		}

		vec3 p = vec3(0, 0, 1);

		float dist = distanceHyper(p, moveVec);
		vec3 v = vSectionHyper(p, moveVec, dist);

		//m1 point
		vec3 m1 = p * cosh(dist / 4) + v * sinh(dist / 4);
		//m2 point
		vec3 m2 = p * cosh(dist * 3 / 4) + v * sinh(dist * 3 / 4);

		for (size_t i = 0; i < VERTICES_NUM; i++)
		{				
			graph.vertices[i] = mirrorHyper(graph.vertices[i], m1);
			graph.vertices[i] = mirrorHyper(graph.vertices[i], m2);
		}
	}
}

vec3 Fe(float dist, vec3 startP, vec3 endP) {
	float F;
	vec3 vecF;
	F = pow((dist - 0.4), 3) * 10000;
	vecF = vSectionHyper(startP, endP, dist);
	vecF = vecF * F;
	return vecF;
}

vec3 Fn(float dist, vec3 startP, vec3 endP) {
	float F;
	vec3 vecF;
	F = (log(dist) - 1.6)*100;
	if (F > 0) F = 0;
	vecF= vSectionHyper(startP, endP, dist);
	vecF = vecF * F;
	return vecF;
}

vec3 Fo(vec3 startP) {
	float F;
	vec3 vecF;
	float dist = distanceHyper(startP, vec3(0, 0, 1));
	if (!(dist > FLT_MIN)) {
		return vec3(0, 0, 0);
	}
	F = dist * 4000; 
	vecF = vSectionHyper(startP, vec3(0, 0, 1), dist);
	vecF=vecF* F;
	return vecF;
}


long startTime = 0;
long previousTime;
bool newStartTime=false;
bool firstStart = false;

void dinSim(double dt) {
	for (int i = 0; i < VERTICES_NUM; i++) {
		vec3 vi = graph.verticesV[i];
		vec3 Fi = vec3(0,0,0);

		for (int j = 0; j < VERTICES_NUM; j++) {
			if (i != j) {
				for (size_t z = 0; z < graph.lines.size(); z++) {
					if (((graph.lines[z].vertex1 == &graph.vertices[i] && graph.lines[z].vertex2 == &graph.vertices[j])
						|| (graph.lines[z].vertex1 == &graph.vertices[j] && graph.lines[z].vertex2 == &graph.vertices[i]))) {

						if (graph.lines[z].used) {
							vec3 newF = Fe(distanceHyper(graph.vertices[i], graph.vertices[j]), graph.vertices[i], graph.vertices[j]);
							if (abs(newF.x) > FLT_MIN && abs(newF.y) > FLT_MIN && abs(newF.z) > FLT_MIN) {
								Fi = Fi + newF;
							}
						}
						else { 
							vec3 newF = Fn(distanceHyper(graph.vertices[i], graph.vertices[j]), graph.vertices[i], graph.vertices[j]);
							if (abs(newF.x) > FLT_MIN && abs(newF.y) > FLT_MIN && abs(newF.z) > FLT_MIN) {
								Fi = Fi + newF;
							} 
						}
					}
				}
			}
		}
		Fi = Fi + Fo(graph.vertices[i]);

		float p = pow(length(vi), 0.8)*40;
		
		vec3 frictionF = vi *p;
		Fi = Fi - frictionF;
	
		if (abs(Fi.x * dt) >= FLT_MIN && abs(Fi.y * dt) >= FLT_MIN && abs(Fi.z * dt) >= FLT_MIN) {
			if (abs(vi.x) < FLT_MIN || abs(vi.y) < FLT_MIN || abs(vi.z) < FLT_MIN) {
				vi = Fi * dt;
				vi = normalize(vi);
			}
			else {
				vi = vi + Fi * dt;
				vi = normalize(vi);
			}
		}
		else if (abs(vi.x) < FLT_MIN || abs(vi.y) < FLT_MIN || abs(vi.z) < FLT_MIN) {
			vi.x = 0;	vi.y = 0;	vi.z = 0;
		}

		float viLength = length(vi);
		float dist = viLength*dt;

		if (!(dist > FLT_MIN)) {
			graph.verticesV[i] = vi;
			continue;
		} 
		
		vec3 newPos =rSectionHyper(graph.vertices[i], vi, dist);
		newPos.z = sqrt(1 + newPos.x * newPos.x + newPos.y * newPos.y);		

		if (abs(newPos.x) > FLT_MIN && abs(newPos.y) > FLT_MIN && abs(newPos.z) > FLT_MIN) {
			vec3 sectionPoint = rSectionHyper(graph.vertices[i], vi, dist * 2);
			graph.vertices[i] = newPos;
			vec3 newV = vSectionHyper(newPos, sectionPoint, distanceHyper(newPos, sectionPoint)) * viLength;
			if (abs(newV.x) > FLT_MIN && abs(newV.y) > FLT_MIN && abs(newV.z) > FLT_MIN) {
				graph.verticesV[i] = newV;
			}
			else graph.verticesV[i] = vec3(0, 0, 0);
		}
		glutPostRedisplay();
	}
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		KMeans();
		glutPostRedisplay(); // if SPACE, invalidate display, i.e. redraw
		newStartTime = true;
		firstStart = true;
	}          
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	graphMove(cX, cY);
	
	glutPostRedisplay();
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	vectorStart.x = cX;
	vectorStart.y = cY;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}
	pressedButton = button;
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	if (newStartTime) {
		startTime = time;
		newStartTime = false;
		previousTime = time;
		return;
	}

	//dinSim(time - previousTime);
	double T = 300;
	if (time < startTime+T && firstStart) {
		double dt = (time - previousTime) / (double)150;
		dinSim(dt);
		previousTime = time;
	}
}
