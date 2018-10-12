#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include "kernel.cuh"
#include <deque>
#include <vector>
#include "helper_math.h"
#include "sdf.h"

typedef unsigned int GLuint;
typedef unsigned int uint;

const int numColors = 8;
const float3 colors[numColors] = {  make_float3(.722f, .141f, .447f),
                                    make_float3(.886f, .455f, .173f),
                                    make_float3(.110f, .569f, .478f),
                                    make_float3(.588f, .824f, .161f),
                                    make_float3(.722f, .267f, .506f),
                                    make_float3(.831f, .345f, .031f),
                                    make_float3(.020f, .533f, .431f),
                                    make_float3(.506f, .773f, .027f)};
const float3 sdfColor = make_float3(1.f, 1.f, 1.f);

class ParticleSystem
{
public:
    ParticleSystem(float particleRadius, uint3 gridSize, uint maxParticles, int3 minBounds, int3 maxBounds, int
    iterations, bool precomputation = true);
    ~ParticleSystem();

    void update(float deltaTime);
    void resetGrid();

    void addFluid(int3 ll, int3 ur, float mass, float density, float3 color);
    void addParticleGrid(int3 ll, int3 ur, float mass, bool addJitter);
    void addHorizCloth(int2 ll, int2 ur, float3 spacing, float2 dist, float mass, bool holdEdges);
    void addRope(float3 start, float3 spacing, float dist, int numLinks, float mass, bool constrainStart);
    void addStaticSphere(int3 ll, int3 ur, float spacing);

    void setParticleToAdd(float3 pos, float3 vel, float mass);
    void setFluidToAdd(float3 pos, float3 color, float mass, float density);

    void makePointConstraint(uint index, float3 point);
    void makeDistanceConstraint(uint2 index, float distance);

    // getters
    std::vector<int2> getColorIndex() { return m_colorIndex; }
    std::vector<float4> getColors() { return m_colors; }

    GLuint getCurrentReadBuffer() const { return m_posVbo; }
    GLuint getCurrentReadBufferSdf() const { return m_posVboSdf; }
    uint getNumParticles() const { return m_numParticles; }
    uint getNumParticlesSdf() const { return m_sdfParticles.size(); }
    float getParticleRadius() const { return m_particleRadius; }

    int3 getMinBounds() { return m_minBounds; }
    int3 getMaxBounds() { return m_maxBounds; }

//    float4 mousePos;

private:
    void _init(uint numParticles, uint maxParticles);
    void _finalize();

    GLuint createVBO(uint size);
    void setArray(bool isVboArray, const float *data, int start, int count);

    void addParticle(float4 pos, float4 vel, float mass, float ro, int phase);
    void addParticleMultiple(float *pos, float *vel, float *mass, float *ro, int *phase, int numParticles);
    void addParticles();

    void addFluids();

    void addNewStuff();

    bool m_initialized;

    float m_particleRadius;

    uint m_maxParticles;
    uint m_numParticles;

    // GPU data
    float *m_dSortedPos;
    float *m_dSortedW;
    int   *m_dSortedPhase;

    // grid data for sorting method
    uint  *m_dGridParticleHash; // grid hash value for each particle
    uint  *m_dGridParticleIndex;// particle index for each particle
    uint  *m_dCellStart;        // index of start of each cell in sorted list
    uint  *m_dCellEnd;          // index of end of cell

    uint   m_gridSortBits;

    // vertex buffer object for particle positions
    GLuint   m_posVbo;

    // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *m_cuda_posvbo_resource;

    // params
    SimParams m_params;
    uint3 m_gridSize;
    uint m_numGridCells;

    // phase number for rigid bodies
    int m_rigidIndex;

    // to store particles that need to be added
    std::deque<float4> m_particlesToAdd;
    std::deque<float4> m_fluidsToAdd;

    // particle colors
    std::vector<int2> m_colorIndex;
    std::vector<float4> m_colors;

    // scene boundaries
    int3 m_minBounds;
    int3 m_maxBounds;

    uint m_solverIterations;
    
    
    // *************************
    // thesis modifications
    // *************************
public:
    void addSDF(SignedDistanceField sdf);
    void prepareScene();
    
    void addDeformableCube(int3 position, float mass, bool addJitter);
    
private:
    void computeSDFSurfaces();
    void addSDFParticles();

    void alignToGrid(float3 &min, float3 &max);
    
    void generateParticlesLocal();
    
    bool m_precomputation;
    
    GLuint m_posVboSdf;
    
    float *m_dSortedPosSdf;
    
    uint *m_dGridParticleHashSdf;
    uint *m_dGridParticleIndexSdf;
    uint *m_dCellStartSdf;
    uint *m_dCellEndSdf;
    
    struct cudaGraphicsResource *m_cuda_posvbosdf_resource;
    
    std::vector<SignedDistanceField> m_sdfs;
    std::vector<float4> m_sdfParticles;

    uint m_numSDFParticles;
    uint m_maxSDFParticles;
    
    uint m_iterations;

    struct Float4Hash
    {
        // we ignore the forth coordinate of the vector here,
        // because it will be equal to 1 in all use cases
        const int p1 = 73856093;
        const int p2 = 19349663;
        const int p3 = 83492791;

        size_t operator() (const float4 p) const
        {
            return static_cast<size_t>(p1 * *reinterpret_cast<const int*>(&p.x) ^
                                       p2 * *reinterpret_cast<const int*>(&p.y) ^
                                       p3 * *reinterpret_cast<const int*>(&p.z)) % (2 * 20000);
        }
    };

    struct Float4Comparator
    {
        bool operator() (const float4 &p1, const float4 &p2) const
        {
            return p1.x == p2.x && p1.y == p2.y && p1.z == p2.z && p1.w == p2.w;
        }
    };
};

#endif // PARTICLESYSTEM_H
