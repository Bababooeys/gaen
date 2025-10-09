use std::f32::consts::PI;

use gaen::*;

struct RotationSpeed(f32);

struct CubeDesc {
    offset: Vec3,
    angle: f32,
    scale: f32,
    rotation: f32,
}

const CUBE_DESC: [CubeDesc; 4] = [
    CubeDesc {
        offset: Vec3::new(-2.0, -2.0, 2.0),
        angle: 10.0,
        scale: 0.7,
        rotation: 0.1,
    },
    CubeDesc {
        offset: Vec3::new(2.0, -2.0, 2.0),
        angle: 50.0,
        scale: 1.3,
        rotation: 0.2,
    },
    CubeDesc {
        offset: Vec3::new(-2.0, 2.0, 2.0),
        angle: 140.0,
        scale: 1.1,
        rotation: 0.3,
    },
    CubeDesc {
        offset: Vec3::new(2.0, 2.0, 2.0),
        angle: 210.0,
        scale: 0.9,
        rotation: 0.4,
    },
];

struct Game;

impl gaen::State for Game {
    fn new(gpu: &Gpu, world: &mut World) -> Self {
        let cube_mesh = Cuboid::new(2, 2, 2).into_mesh(gpu);
        let plane_mesh = Plane::new(16, 16).into_mesh(gpu);

        world.spawn((
            Transform::new(Mat4::IDENTITY),
            RotationSpeed(0.0),
            Material::new(Vec4::ONE),
            plane_mesh,
        ));

        for cube in CUBE_DESC.iter() {
            let matrix = Mat4::from_scale_rotation_translation(
                Vec3::splat(cube.scale),
                Quat::from_axis_angle(cube.offset.normalize(), cube.angle * PI / 180.),
                cube.offset,
            );
            world.spawn((
                Transform::new(matrix),
                RotationSpeed(cube.rotation),
                Material::new(Vec4::new(0.0, 1.0, 0.0, 1.0)),
                cube_mesh.clone(),
            ));
        }

        world.spawn((Light {
            pos: Vec3::new(7.0, -5.0, 10.0),
            color: Vec3::new(0.5, 1.0, 0.5),
            fov: 60.0,
            depth: 1.0..20.0,
        },));

        world.spawn((Light {
            pos: Vec3::new(-5.0, 7.0, 10.0),
            color: Vec3::new(1.0, 0.5, 0.5),
            fov: 45.0,
            depth: 1.0..20.0,
        },));

        Self
    }

    fn update(&mut self, _gpu: &Gpu, world: &mut World) {
        for (_, (transform, rotation_speed)) in
            world.query_mut::<(&mut Transform, &RotationSpeed)>()
        {
            if rotation_speed.0 != 0.0 {
                let rotation = Mat4::from_rotation_x(rotation_speed.0 * PI / 180.);
                transform.matrix *= rotation;
            }
        }
    }
}

fn main() {
    run::<Game>().unwrap();
}
