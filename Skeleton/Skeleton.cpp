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

	void main() {
		gl_Position = vec4(vp.x/vp.z, vp.y/vp.z, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

static const int VERTICES_NUM = 50;
static const double FULLNESS = 0.05;

static const int LINES_NUM = round(VERTICES_NUM * (VERTICES_NUM - 1) / 2 * FULLNESS);

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
public:
	std::vector<vec3> vertices;
	std::vector<vec3> verticesV;
	std::vector<Line> lines;

	Graph() {
		for (int i = 0; i < VERTICES_NUM; i++) {
			verticesV.push_back(vec3(0, 0, 0));
		}
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vaoVertices;	   // virtual world on the GPU
unsigned int vboVertices;		// vertex buffer object
unsigned int vaoLines;
unsigned int vboLines;

int pressedButton;
Graph graph;
vec2 vectorStart;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vaoLines);	// get 1 vao id
	glGenBuffers(1, &vboLines);	// Generate 1 buffer
	
	glGenVertexArrays(1, &vaoVertices);	// get 1 vao id
	glGenBuffers(1, &vboVertices);	// Generate 1 buffer
	
	graph = Graph();
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)


	for (int i = 0; i < VERTICES_NUM; i++) {
		float x = ((((float)(rand() * 2) ) / (RAND_MAX))- 1.0f)*1.5;
		float y = ((((float)(rand() * 2)) / (RAND_MAX)) - 1.0f)*1.5;
		
		graph.vertices.push_back(vec3(x, y, (float)sqrt(1+x*x+y*y)));
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

void drawPoints() {
	glBindVertexArray(vaoVertices);
	glBindBuffer(GL_ARRAY_BUFFER, vboVertices);

	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		graph.vertices.size() * sizeof(vec3),  // # bytes
		&graph.vertices[0],	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		3, GL_FLOAT, GL_FALSE, // three floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed

	glBindVertexArray(vaoVertices);  // Draw call
	glDrawArrays(GL_POINTS, 0 /*startIdx*/, VERTICES_NUM /*# Elements*/);
}

void drawLines() {
	glBindVertexArray(vaoLines);
	glBindBuffer(GL_ARRAY_BUFFER, vboLines);

	std::vector<vec3> usedLines;
	for (size_t i = 0; i < graph.lines.size(); i++)
	{
		if (graph.lines[i].used) {
			usedLines.push_back(*graph.lines[i].vertex1);
			usedLines.push_back(*graph.lines[i].vertex2);
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

	glBindVertexArray(vaoLines);  // Draw call
	glDrawArrays(GL_LINES, 0 /*startIdx*/, LINES_NUM * 2 /*# Elements*/);
}

void drawGraph() {
	// Set color to (0, 1, 0) = green
	int color = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(color, 1.0f, 0.0f, 1.0f); // 3 floats
	drawPoints();

	glUniform3f(color, 1.0f, 1.0f, 0.0f); //lines are yellow
	drawLines();
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

	drawGraph(); // gráf rajzolása

	glutSwapBuffers(); // exchange buffers for double buffering
}

void KMeans() { //ennek kell még valami hogy többször is lehessen egy más után szóval ha valahol FLT_MIN alá megy valami akkor inkább ne is történjen semmi

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
		if (!(abs(center.x) >= FLT_MIN) || !(abs(center.y) >= FLT_MIN)) {
			printf("problem\n");
		}
		center = center / sqrt(1 - center.x * center.x - center.y * center.y);
		
		if (abs(center.x) >= FLT_MIN && abs(center.y) >= FLT_MIN && abs(center.z) >= FLT_MIN) {
			graph.vertices[i] = center;
		}

		//printf("%lf, %lf, %lf = %lf\n", center.x, center.y, center.z, (center.x * center.x + center.y * center.y - center.z * center.z));
	}
}

float distanceHyper(vec3 p, vec3 q) {
	//printf("%lf, %lf\n", q.x, p.x);
	//printf("%lf\n", p.x * q.x + p.y * q.y - p.z * q.z);
	return acosh((-1)*(p.x*q.x + p.y*q.y - p.z*q.z));
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

		//printf("%lf", FLT_MIN);
		//printf("%lf, %lf, %lf\t = %lf\n", moveVec.x, moveVec.y, moveVec.z);
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
			//printf("%d. %lf, %lf, %lf\t = %lf\n", i, graph.vertices[i].x, graph.vertices[i].y, graph.vertices[i].z, (graph.vertices[i].x * graph.vertices[i].x + graph.vertices[i].y * graph.vertices[i].y - graph.vertices[i].z * graph.vertices[i].z));
			/*temp = mirrorHyper(temp, m1);
			temp = mirrorHyper(temp, m2);

			if (!(abs(temp.x) < FLT_MIN || abs(temp.y) < FLT_MIN || abs(temp.z) < FLT_MIN))
				graph.vertices[i] = temp;
			else printf("baj\n");*/


			//printf("%lf, %lf, %lf\t = %lf\n", graph.vertices[i].x, graph.vertices[i].y, graph.vertices[i].z, (graph.vertices[i].x * graph.vertices[i].x + graph.vertices[i].y * graph.vertices[i].y - graph.vertices[i].z * graph.vertices[i].z));
		}
	}
}

vec3 Fe(float dist, vec3 startP, vec3 endP) {
	float F;
	vec3 vecF;
	F = pow((dist - 0.2), 3) * 200;
	if (F > 100) F = 100;
	vecF = vSectionHyper(startP, endP, dist);
	//printf("dist: %lf %lf %lf %lf \n",dist, vecF.x, vecF.y, vecF.z);
	vecF = vecF * F;
	return vecF;
}

vec3 Fn(float dist, vec3 startP, vec3 endP) {
	float F;
	vec3 vecF;
	F = log(dist) - 1;
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
	
	F = dist * 30;
	vecF = vSectionHyper(startP, vec3(0, 0, 1), dist);
	vecF=vecF* F;
	return vecF;
}


long startTime = 0;
long previousTime;
bool newStartTime=false;
bool firstStart = false;

void dinSim(double dt) {
	int num = 0;
	for (int i = 0; i < VERTICES_NUM; i++) {
		vec3 vi = graph.verticesV[i];
		//printf("...............................................\n");
		//printf("[%d]. %f, %f, %f\n", i, vi.x, vi.y, vi.z);
		
		vec3 Fi = vec3(0,0,0);
		//printf("Fi: [%d]. %f, %f, %f\n", i, Fi.x, Fi.y, Fi.z);
		for (int j = 0; j < VERTICES_NUM; j++) {
			if (i != j) {
				for (size_t z = 0; z < graph.lines.size(); z++) {
					if (((graph.lines[z].vertex1 == &graph.vertices[i] && graph.lines[z].vertex2 == &graph.vertices[j])
						|| (graph.lines[z].vertex1 == &graph.vertices[j] && graph.lines[z].vertex2 == &graph.vertices[i]))) {

						if (graph.lines[z].used) {
							vec3 newF = Fe(distanceHyper(graph.vertices[i], graph.vertices[j]), graph.vertices[i], graph.vertices[j]);
							//printf("%lf, %lf, %lf\n",  newF.x, newF.y, newF.z);
							if (abs(newF.x) > FLT_MIN && abs(newF.y) > FLT_MIN && abs(newF.z) > FLT_MIN) {
								Fi = Fi + newF;
							}
						}
						else { 
							vec3 newF = Fn(distanceHyper(graph.vertices[i], graph.vertices[j]), graph.vertices[i], graph.vertices[j]);
							//printf("%lf, %lf, %lf\n", newF.x, newF.y, newF.z);
							if (abs(newF.x) > FLT_MIN && abs(newF.y) > FLT_MIN && abs(newF.z) > FLT_MIN) {
								Fi = Fi + newF;
							} 
						}
					}
				}
			}
		}
		//if (length(vi) < FLT_MIN) vi = vec3(0, 0, 0);
		//printf("%lf %lf %lf \n", vi.x, vi.y, vi.z);
		Fi = Fi + Fo(graph.vertices[i]);
		
		printf("%lf \n", length(vi));
		vec3 frictionF = vi * pow(length(vi), 100);
		/*if (length(vi) > 1) {
			frictionF = vi * pow(length(vi), 100);
		} else frictionF = vi * length(vi)*2;*/
		//vec3 frictionF = vi * length(vi)*50;
		//frictionF = normalize(frictionF);
		if(length(frictionF) > FLT_MIN) Fi = Fi - frictionF;


		/*printf("%lf %lf %lf \n", Fi.x, Fi.y, Fi.z);
		printf("dt: %lf %lf %lf \n", Fi.x*dt, Fi.y*dt, Fi.z*dt);
		printf("vi elott: %lf %lf %lf \n", vi.x, vi.y, vi.z);*/
		//Fi = normalize(Fi);

		if (abs(Fi.x * dt) >= FLT_MIN && abs(Fi.y * dt) >= FLT_MIN && abs(Fi.z * dt) >= FLT_MIN) {
			if (abs(vi.x) < FLT_MIN || abs(vi.y) < FLT_MIN || abs(vi.z) < FLT_MIN) {
				//printf("baj\n");
				vi = Fi * dt;
				vi = normalize(vi);
			}
			else {
				vi = vi + Fi * dt;
				vi = normalize(vi);
				//printf("x\n");
			}
		}
		else if (abs(vi.x) < FLT_MIN || abs(vi.y) < FLT_MIN || abs(vi.z) < FLT_MIN) {
			//printf("baj2\n");
			vi.x = 0;
			vi.y = 0;
			vi.z = 0;
			
		}
		
		//printf("vi utan: %lf %lf %lf \n", vi.x, vi.y, vi.z);
		
		float viLength = length(vi);
		//printf("viL: %lf\n", viLength);
		float dist = viLength*dt;
		//printf("%lf\n", dist);
		if (!(dist > FLT_MIN)) {
			//graph.verticesV[i] = vec3(0,0,0);
			graph.verticesV[i] = vi;
			//printf("%lf\n", dist);
			continue;
		} 
		//printf("%lf\n", dist);
		
		vec3 newPos =rSectionHyper(graph.vertices[i], vi, dist);
		newPos.z = sqrt(1 + newPos.x * newPos.x + newPos.y * newPos.y);
		//printf("Newpos: %lf %lf %lf = %lf\n", newPos.x, newPos.y, newPos.z, newPos.x* newPos.x + newPos.y* newPos.y - newPos.z* newPos.z);
		//printf("---------------------------------------------------------------------\n");
		

		if (abs(newPos.x) > FLT_MIN && abs(newPos.y) > FLT_MIN && abs(newPos.z) > FLT_MIN) {
			//printf("siker\n");
			vec3 sectionPoint = rSectionHyper(graph.vertices[i], vi, dist*2);
			graph.vertices[i] = newPos;
			vec3 newV = vSectionHyper(newPos, sectionPoint, distanceHyper(newPos, sectionPoint)) * viLength;
			if (abs(newV.x) > FLT_MIN && abs(newV.y) > FLT_MIN && abs(newV.z) > FLT_MIN){
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
		glutPostRedisplay();
		newStartTime = true;
		firstStart = true;
	}          // if d, invalidate display, i.e. redraw
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
	
	
	//printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
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

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
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
	double T = 10000;
	if (time < startTime+T && firstStart) {
		double dt = (time - previousTime) / (double)1000;
		dinSim(dt);
		previousTime = time;
		glutPostRedisplay();
	}

}
