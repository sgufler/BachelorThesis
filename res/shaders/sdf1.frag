in vec3 nearPlanePos;

uniform vec4 color;
uniform vec3 eye;
uniform mat4 VP;

const float epsilon = 0.001;
const int maxSteps = 64;

float sd_sphere(vec3 pos)
{
    const float radius = 3.0;
    const vec3 center = vec3(4, 0, 0);

    return length(pos - center) - radius;
}

float sd_scene(vec3 pos)
{
    return sd_sphere(pos);
}

vec3 phong(
  in vec3 pt,
  in vec3 prp,
  in vec3 normal,
  in vec3 light,
  in vec3 color,
  in float spec,
  in vec3 ambLight)
{
   vec3 lightv=normalize(light-pt);
   float diffuse=dot(normal,lightv);
   vec3 refl=-reflect(lightv,normal);
   vec3 viewv=normalize(prp-pt);
   float specular=pow(max(dot(refl,viewv),0.0),spec);
   return (max(diffuse,0.0)+ambLight)*color+specular;
}

vec3 normal(in vec3 p)
{
  //tetrahedron normal
  const float n_er=0.01;
  float v1=sd_scene(vec3(p.x+n_er,p.y-n_er,p.z-n_er));
  float v2=sd_scene(vec3(p.x-n_er,p.y-n_er,p.z+n_er));
  float v3=sd_scene(vec3(p.x-n_er,p.y+n_er,p.z-n_er));
  float v4=sd_scene(vec3(p.x+n_er,p.y+n_er,p.z+n_er));
  return normalize(vec3(v4+v1-v3-v2,v3+v4-v1-v2,v2+v4-v3-v1));
}

void main() 
{
    vec3 origin = eye;
    vec3 dir = normalize(nearPlanePos - origin);

    float currentDistance = 0.0;
    bool hit = false;

    for (int i = 0; i < maxSteps; i++)
    {
        float dist = sd_scene(origin + dir * currentDistance);
        if (dist > epsilon)
        {
            currentDistance += dist;
        }
        else
        {
            hit = true;
            break;
        }
    }

    if (hit)
    {
        const float spec=8.0;
        const vec3 ambLight = vec3(0.1,0.1,0.1);
        vec3 light = origin + vec3(5.0, 0, 5.0);
        vec3 p = origin + dir * currentDistance;
        vec3 n = normal(p);
        float dist = sd_scene(origin + dir * currentDistance);

        vec3 cf = phong(p, origin, n, light, vec3(color), spec, ambLight);

        gl_FragColor = vec4(cf, 1.0);
        vec4 tmp = VP * vec4(origin + dir * currentDistance, 1.0);
        gl_FragDepth = tmp.z / tmp.w * 0.5 + 0.5;
    }
    else
    {
        discard;
    }
}
