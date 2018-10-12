#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <vector>
#include <vector_types.h>

class SignedDistanceField
{
public:
    SignedDistanceField(std::function<float(glm::vec3)>);

    SignedDistanceField(std::function<float(glm::vec3)>, glm::vec3 offset);

    float evaluate(glm::vec3 p);

    glm::vec3 gradient(glm::vec3 x);

    SignedDistanceField intersect(SignedDistanceField other);

    SignedDistanceField unite(SignedDistanceField other);

    SignedDistanceField difference(SignedDistanceField other);

private:

    std::function<float(glm::vec3)> function;
    glm::vec3 offset;
};
