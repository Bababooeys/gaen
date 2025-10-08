use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::Gpu;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [i8; 4],
    pub normal: [i8; 4],
}

impl Vertex {
    pub const fn new(pos: [i8; 3], nor: [i8; 3]) -> Self {
        Self {
            pos: [pos[0], pos[1], pos[2], 1],
            normal: [nor[0], nor[1], nor[2], 0],
        }
    }
}

pub struct Cuboid {
    pub width: i8,
    pub height: i8,
    pub depth: i8,
}

impl Cuboid {
    pub const fn new(width: i8, height: i8, depth: i8) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    const fn vertices(half_x: i8, half_y: i8, half_z: i8) -> [Vertex; 24] {
        [
            // top (0, 0, 1)
            Vertex::new([-half_x, -half_y, half_z], [0, 0, 1]),
            Vertex::new([half_x, -half_y, half_z], [0, 0, 1]),
            Vertex::new([half_x, half_y, half_z], [0, 0, 1]),
            Vertex::new([-half_x, half_y, half_z], [0, 0, 1]),
            // bottom (0, 0, -1)
            Vertex::new([-half_x, half_y, -half_z], [0, 0, -1]),
            Vertex::new([half_x, half_y, -half_z], [0, 0, -1]),
            Vertex::new([half_x, -half_y, -half_z], [0, 0, -1]),
            Vertex::new([-half_x, -half_y, -half_z], [0, 0, -1]),
            // right (1, 0, 0)
            Vertex::new([half_x, -half_y, -half_z], [1, 0, 0]),
            Vertex::new([half_x, half_y, -half_z], [1, 0, 0]),
            Vertex::new([half_x, half_y, half_z], [1, 0, 0]),
            Vertex::new([half_x, -half_y, half_z], [1, 0, 0]),
            // left (-1, 0, 0)
            Vertex::new([-half_x, -half_y, half_z], [-1, 0, 0]),
            Vertex::new([-half_x, half_y, half_z], [-1, 0, 0]),
            Vertex::new([-half_x, half_y, -half_z], [-1, 0, 0]),
            Vertex::new([-half_x, -half_y, -half_z], [-1, 0, 0]),
            // front (0, 1, 0)
            Vertex::new([half_x, half_y, -half_z], [0, 1, 0]),
            Vertex::new([-half_x, half_y, -half_z], [0, 1, 0]),
            Vertex::new([-half_x, half_y, half_z], [0, 1, 0]),
            Vertex::new([half_x, half_y, half_z], [0, 1, 0]),
            // back (0, -1, 0)
            Vertex::new([half_x, -half_y, half_z], [0, -1, 0]),
            Vertex::new([-half_x, -half_y, half_z], [0, -1, 0]),
            Vertex::new([-half_x, -half_y, -half_z], [0, -1, 0]),
            Vertex::new([half_x, -half_y, -half_z], [0, -1, 0]),
        ]
    }

    const INDICES: [u16; 36] = [
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    pub fn into_mesh(self, gpu: &Gpu) -> Mesh {
        Mesh::new(
            gpu,
            &Self::vertices(self.width / 2, self.height / 2, self.depth / 2),
            &Self::INDICES,
        )
    }
}

pub struct Plane {
    pub width: i8,
    pub height: i8,
}

impl Plane {
    pub const fn new(width: i8, height: i8) -> Self {
        Self { width, height }
    }

    const fn vertices(half_x: i8, half_y: i8) -> [Vertex; 4] {
        [
            Vertex::new([half_x, -half_y, 0], [0, 0, 1]),
            Vertex::new([half_x, half_y, 0], [0, 0, 1]),
            Vertex::new([-half_x, -half_y, 0], [0, 0, 1]),
            Vertex::new([-half_x, half_y, 0], [0, 0, 1]),
        ]
    }

    const INDICES: [u16; 6] = [0, 1, 2, 2, 1, 3];

    pub fn into_mesh(self, gpu: &Gpu) -> Mesh {
        Mesh::new(
            gpu,
            &Self::vertices(self.width / 2, self.height / 2),
            &Self::INDICES,
        )
    }
}

#[derive(Clone)]
pub struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    index_count: usize,
}

impl Mesh {
    pub fn new(gpu: &Gpu, vertices: &[Vertex], indices: &[u16]) -> Self {
        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuMesh Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuMesh Index Buffer"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let index_format = wgpu::IndexFormat::Uint16;
        let index_count = indices.len();
        Self {
            vertex_buffer,
            index_buffer,
            index_format,
            index_count,
        }
    }

    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass) {
        pass.set_index_buffer(self.index_buffer.slice(..), self.index_format);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw_indexed(0..self.index_count as u32, 0, 0..1);
    }
}
