#define GLM_ENABLE_EXPERIMENTAL

#include <algorithm>
// #include <gtx/hash.hpp>
#include <unordered_set>
#include <iostream>
#include "sdf.h"

#define EPSILON 0.001f
#define PARTICLE_DIAM 0.25f

SignedDistanceField::SignedDistanceField(std::function<float(glm::vec3)> function)
        : function(function), offset(glm::vec3())
{

}

SignedDistanceField::SignedDistanceField(std::function<float(glm::vec3)> function, glm::vec3 offset)
        : function(function), offset(offset)
{

}

float SignedDistanceField::evaluate(glm::vec3 p)
{
    return function(p - offset);
}

glm::vec3 SignedDistanceField::gradient(glm::vec3 p)
{
    glm::vec3 h = p * (float) sqrt(EPSILON);
    float dx = evaluate(glm::vec3(p.x + 0.5f * h.x, p.y, p.z)) - evaluate(glm::vec3(p.x - 0.5f * h.x, p.y, p.z));
    float dy = evaluate(glm::vec3(p.x, p.y + 0.5f * h.y, p.z)) - evaluate(glm::vec3(p.x, p.y - 0.5f * h.y, p.z));
    float dz = evaluate(glm::vec3(p.x, p.y, p.z + 0.5f * h.z)) - evaluate(glm::vec3(p.x, p.y, p.z - 0.5f * h.z));

    return glm::vec3(dx, dy, dz);
}


SignedDistanceField SignedDistanceField::intersect(SignedDistanceField other)
{
    std::function<float(glm::vec3)> &f1 = this->function;
    std::function<float(glm::vec3)> &f2 = other.function;

    return {[f1, f2](glm::vec3 p) -> float { return glm::min(f1(p), f2(p)); }};
}

SignedDistanceField SignedDistanceField::unite(SignedDistanceField other)
{
    std::function<float(glm::vec3)> &f1 = this->function;
    std::function<float(glm::vec3)> &f2 = other.function;

    return {[f1, f2](glm::vec3 p) -> float { return glm::max(f1(p), f2(p)); }};
}

SignedDistanceField SignedDistanceField::difference(SignedDistanceField other)
{
    std::function<float(glm::vec3)> &f1 = this->function;
    std::function<float(glm::vec3)> &f2 = other.function;

    return {[f1, f2](glm::vec3 p) -> float { return glm::max(f1(p), -f2(p)); }};
}
