#ifndef RENDERER_H
#define RENDERER_H

#include <glm/glm.hpp>
#include <vector_types.h>
#include <vector>

typedef unsigned int GLuint;
typedef unsigned int uint;
class ActionCamera;
class QMouseEvent;
class QKeyEvent;

class Renderer
{
public:
    Renderer(int3 minBounds, int3 maxBounds);
    ~Renderer();

    void createVAO(GLuint vbo, float radius);
    void createSdfVAO(GLuint vbo, int numParticles);

    void setVBO(GLuint vbo, uint numParticles);
    void render(std::vector<int2> colorIndices, std::vector<float4> colors);

    float4 raycast2XYPlane(float x, float y);
    float3 getDir(float x, float y);
    float3 getEye();

//    void mousePressed(QMouseEvent *e);
    void mouseMoved(QMouseEvent *, float deltaX, float deltaY);
//    void mouseScrolled(QWheelEvent *e);

    void keyPressed(QKeyEvent* e);
    void keyReleased(QKeyEvent* e);

    void update(float secs);
    void setNumParticlesSdf(int num);

    void resize(int w, int h);
    
    void setSdfSceneID(int id);

private:
    void _initGL();
    GLuint _compileProgram(const char *vertex_file_path, const char *fragment_file_path);

    void _buildGrid(int3 minBounds, int3 maxBounds);
    void _setGridBuffer(float *data, int memsize);

    void createRenderSdfVAO();

    GLuint m_program;
    GLuint m_vbo;
    GLuint m_vao;

    GLuint m_vboGrid;
    GLuint m_vaoGrid;
    int m_numGridVerts;
    
    GLuint m_vboSdf;
    GLuint m_vaoSdf;
    int m_numParticlesSdf;
    
    GLuint m_programSdf;
    int m_sdfSceneID;
    GLuint m_vboRenderSdf;
    GLuint m_vaoRenderSdf;
    
    bool m_toggleRenderSdf;

    ActionCamera *m_camera;
    glm::ivec2 m_screenSize;

    float m_particleRadius;
    int m_wsadeq;

};

#endif // RENDERER_H
