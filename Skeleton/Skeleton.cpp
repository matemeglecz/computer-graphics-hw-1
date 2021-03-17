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
// Nev    : 
// Neptun : 
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

static const int NUM_OF_VERTICES = 50;
static const double FULLNESS = 0.05;

static const int NUM_OF_LINES = round(NUM_OF_VERTICES * (NUM_OF_VERTICES - 1) / 2 * FULLNESS);

class Line {
public:
	vec3 vertex1;
	vec3 vertex2;
	bool used;

	Line(vec3 v1, vec3 v2, bool u= false) {
		vertex1 = v1;
		vertex2 = v2;
		used = u;
	}
};

class Graph {
public:
	std::vector<vec3> vertices;
	std::vector<Line> lines;
};

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vaoVertices;	   // virtual world on the GPU
unsigned int vboVertices;		// vertex buffer object
unsigned int vaoLines;
unsigned int vboLines;

Graph graph;







// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	std::vector<vec3> vertices=graph.vertices;

	glGenVertexArrays(1, &vaoVertices);	// get 1 vao id
	glBindVertexArray(vaoVertices);		// make it active

	glGenBuffers(1, &vboVertices);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vboVertices);
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)

	
	for (int i = 0; i < NUM_OF_VERTICES; i++) {
		float x = ((((float)(rand() * 2) ) / (RAND_MAX))- 1.0f)*1.5;
		float y = ((((float)(rand() * 2)) / (RAND_MAX)) - 1.0f)*1.5;
		/*float x = rand() % 100 * 0.01;
		float y = rand() % 100 * 0.01;*/

		//x = ((((float)rand()) / (float)RAND_MAX) * 2 - 1.0f)*2;
		//y = ((((float)rand()) / (float)RAND_MAX) * 2 - 1.0f)*2;

		//float x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		//float y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		//printf("%lf - %lf", x, y);

		
		
		/*if ((int)rand() % 2 == 0) {
			x = (x * (-1));
		}
		if ((int)rand() % 2 == 0) {
			y = (y * (-1));
		}*/
		
		vertices.push_back(vec3(x, y, (float)sqrt(1+x*x+y*y)));
		printf("%lf - %lf - %lf\n", x, y, (float)sqrt(1 + x * x + y * y));
		printf("%lf - %lf\n", x/ (float)sqrt(1 + x * x + y * y), y/ (float)sqrt(1 + x * x + y * y));

	}

	//vertices.push_back(vec3(3 ,0, (float)sqrt(1 + 3 * 3 + 0  )));
	
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		vertices.size()*sizeof(vec3),  // # bytes
		&vertices[0],	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		3, GL_FLOAT, GL_FALSE, // three floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed


	//lines generate
	for (int i = 0; i < NUM_OF_VERTICES; i++) {
		for (int j = 0; j < i; j++) {
			graph.lines.push_back(Line(vertices[i], vertices[j]));
		}
	}

	std::vector<vec3> lines;

	printf("\n%d", graph.lines.size());

	srand(154363445326234556);
	

	for (int i = 0; i < NUM_OF_LINES; i++) {
		bool success = false;
		while (!success) {
			int randidx = rand() % (NUM_OF_VERTICES * (NUM_OF_VERTICES-1)/2);
			//printf("%d\n", randidx);
			if (!graph.lines[randidx].used) {
				graph.lines[randidx].used = true;
				success = true;
				
				lines.push_back(graph.lines[randidx].vertex1);
				lines.push_back(graph.lines[randidx].vertex2);
			}
		}
	}
	
	printf("\n%d", lines.size());

	glGenVertexArrays(1, &vaoLines);	// get 1 vao id
	glBindVertexArray(vaoLines);		// make it active

	glGenBuffers(1, &vboLines);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vboLines);

	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		lines.size() * sizeof(vec3),  // # bytes
		&lines[0],	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		3, GL_FLOAT, GL_FALSE, // three floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed


	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glPointSize(7.0f);
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int color = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(color, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	int location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	glBindVertexArray(vaoVertices);  // Draw call
	glDrawArrays(GL_POINTS, 0 /*startIdx*/, NUM_OF_VERTICES /*# Elements*/);

	glUniform3f(color, 1.0f, 1.0f, 0.0f); //lines are yellow

	glBindVertexArray(vaoLines);  // Draw call
	glDrawArrays(GL_LINES, 0 /*startIdx*/, NUM_OF_LINES*2 /*# Elements*/);

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
