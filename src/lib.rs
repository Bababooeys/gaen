use std::{
    f32::consts::{FRAC_PI_4, PI},
    ops::Range,
    sync::Arc,
};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

mod mesh;

pub use mesh::*;

struct Entity {
    world: glam::Mat4,
    rotation_speed: f32,
    color: glam::Vec4,
    mesh: Mesh,
}

struct Light {
    pos: glam::Vec3,
    color: glam::Vec3,
    fov: f32,
    depth: Range<f32>,
    target_view: wgpu::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightRaw {
    proj: [[f32; 4]; 4],
    pos: [f32; 4],
    color: [f32; 4],
}

impl Light {
    fn to_raw(&self) -> LightRaw {
        let view = glam::Mat4::look_at_rh(self.pos, glam::Vec3::ZERO, glam::Vec3::Z);
        let projection =
            glam::Mat4::perspective_rh(self.fov * PI / 180., 1.0, self.depth.start, self.depth.end);
        let view_proj = projection * view;
        LightRaw {
            proj: view_proj.to_cols_array_2d(),
            pos: [self.pos.x, self.pos.y, self.pos.z, 1.0],
            color: [self.color.x, self.color.y, self.color.z, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalUniforms {
    proj: [[f32; 4]; 4],
    num_lights: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EntityUniforms {
    model: [[f32; 4]; 4],
    color: [f32; 4],
}

struct Pass {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
}

struct Example {
    entities: Vec<Entity>,
    lights: Vec<Light>,
    lights_are_dirty: bool,
    shadow_pass: Pass,
    forward_pass: Pass,
    forward_depth: wgpu::TextureView,
    entity_bind_group: wgpu::BindGroup,
    light_storage_buf: wgpu::Buffer,
    entity_uniform_buf: wgpu::Buffer,
}

impl Example {
    const FEATURES: wgpu::Features = wgpu::Features::DEPTH_CLIP_CONTROL;
    const MAX_LIGHTS: usize = 10;
    const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    const SHADOW_SIZE: wgpu::Extent3d = wgpu::Extent3d {
        width: 512,
        height: 512,
        depth_or_array_layers: Self::MAX_LIGHTS as u32,
    };
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    fn generate_matrix(aspect_ratio: f32) -> glam::Mat4 {
        let projection = glam::Mat4::perspective_rh(FRAC_PI_4, aspect_ratio, 1.0, 20.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(3.0f32, -10.0, 6.0),
            glam::Vec3::new(0f32, 0.0, 0.0),
            glam::Vec3::Z,
        );
        projection * view
    }

    fn create_depth_texture(
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
    ) -> wgpu::TextureView {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
            view_formats: &[],
        });

        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn new(adapter: &wgpu::Adapter, gpu: &Gpu) -> Self {
        let supports_storage_resources = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::VERTEX_STORAGE)
            && gpu.device.limits().max_storage_buffers_per_shader_stage > 0;

        // Create the vertex and index buffers
        let vertex_size = size_of::<Vertex>();
        let cube_mesh = Cuboid::new(2, 2, 2).into_mesh(gpu);
        let plane_mesh = Plane::new(16, 16).into_mesh(gpu);

        struct CubeDesc {
            offset: glam::Vec3,
            angle: f32,
            scale: f32,
            rotation: f32,
        }
        let cube_descs = [
            CubeDesc {
                offset: glam::Vec3::new(-2.0, -2.0, 2.0),
                angle: 10.0,
                scale: 0.7,
                rotation: 0.1,
            },
            CubeDesc {
                offset: glam::Vec3::new(2.0, -2.0, 2.0),
                angle: 50.0,
                scale: 1.3,
                rotation: 0.2,
            },
            CubeDesc {
                offset: glam::Vec3::new(-2.0, 2.0, 2.0),
                angle: 140.0,
                scale: 1.1,
                rotation: 0.3,
            },
            CubeDesc {
                offset: glam::Vec3::new(2.0, 2.0, 2.0),
                angle: 210.0,
                scale: 0.9,
                rotation: 0.4,
            },
        ];

        let entity_uniform_size = size_of::<EntityUniforms>() as wgpu::BufferAddress;
        let num_entities = 1 + cube_descs.len() as wgpu::BufferAddress;
        // Make the `uniform_alignment` >= `entity_uniform_size` and aligned to `min_uniform_buffer_offset_alignment`.
        let uniform_alignment = {
            let alignment =
                gpu.device.limits().min_uniform_buffer_offset_alignment as wgpu::BufferAddress;
            wgpu::util::align_to(entity_uniform_size, alignment)
        };
        // Note: dynamic uniform offsets also have to be aligned to `Limits::min_uniform_buffer_offset_alignment`.
        let entity_uniform_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: num_entities * uniform_alignment,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut entities = vec![{
            Entity {
                world: glam::Mat4::IDENTITY,
                rotation_speed: 0.0,
                color: glam::Vec4::ONE,
                mesh: plane_mesh,
            }
        }];

        for cube in cube_descs.iter() {
            let world = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::splat(cube.scale),
                glam::Quat::from_axis_angle(cube.offset.normalize(), cube.angle * PI / 180.),
                cube.offset,
            );
            entities.push(Entity {
                world,
                rotation_speed: cube.rotation,
                color: glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
                mesh: cube_mesh.clone(),
            });
        }

        let local_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: wgpu::BufferSize::new(entity_uniform_size),
                        },
                        count: None,
                    }],
                    label: None,
                });
        let entity_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &local_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &entity_uniform_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(entity_uniform_size),
                }),
            }],
            label: None,
        });

        // Create other resources
        let shadow_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let shadow_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            size: Self::SHADOW_SIZE,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut shadow_target_views = (0..2)
            .map(|i| {
                Some(shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow"),
                    format: None,
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    usage: None,
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                }))
            })
            .collect::<Vec<_>>();

        let lights = vec![
            Light {
                pos: glam::Vec3::new(7.0, -5.0, 10.0),
                color: glam::Vec3::new(0.5, 1.0, 0.5),
                fov: 60.0,
                depth: 1.0..20.0,
                target_view: shadow_target_views[0].take().unwrap(),
            },
            Light {
                pos: glam::Vec3::new(-5.0, 7.0, 10.0),
                color: glam::Vec3::new(1.0, 0.5, 0.5),
                fov: 45.0,
                depth: 1.0..20.0,
                target_view: shadow_target_views[1].take().unwrap(),
            },
        ];

        let light_uniform_size = (Self::MAX_LIGHTS * size_of::<LightRaw>()) as wgpu::BufferAddress;
        let light_storage_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: light_uniform_size,
            usage: if supports_storage_resources {
                wgpu::BufferUsages::STORAGE
            } else {
                wgpu::BufferUsages::UNIFORM
            } | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_attr = wgpu::vertex_attr_array![0 => Sint8x4, 1 => Sint8x4];
        let vb_desc = wgpu::VertexBufferLayout {
            array_stride: vertex_size as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attr,
        };

        let shader = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let shadow_pass = {
            let uniform_size = size_of::<GlobalUniforms>() as wgpu::BufferAddress;
            // Create pipeline layout
            let bind_group_layout =
                gpu.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0, // global
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(uniform_size),
                            },
                            count: None,
                        }],
                    });
            let pipeline_layout =
                gpu.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("shadow"),
                        bind_group_layouts: &[&bind_group_layout, &local_bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let uniform_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: uniform_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Create bind group
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                }],
                label: None,
            });

            // Create the render pipeline
            let pipeline = gpu
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("shadow"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_bake"),
                        compilation_options: Default::default(),
                        buffers: &[vb_desc.clone()],
                    },
                    fragment: None,
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        unclipped_depth: gpu
                            .device
                            .features()
                            .contains(wgpu::Features::DEPTH_CLIP_CONTROL),
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: Self::SHADOW_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::LessEqual,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState {
                            constant: 2, // corresponds to bilinear filtering
                            slope_scale: 2.0,
                            clamp: 0.0,
                        },
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });

            Pass {
                pipeline,
                bind_group,
                uniform_buf,
            }
        };

        let forward_pass = {
            // Create pipeline layout
            let bind_group_layout =
                gpu.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0, // global
                                visibility: wgpu::ShaderStages::VERTEX
                                    | wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: wgpu::BufferSize::new(
                                        size_of::<GlobalUniforms>() as _,
                                    ),
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1, // lights
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Buffer {
                                    ty: if supports_storage_resources {
                                        wgpu::BufferBindingType::Storage { read_only: true }
                                    } else {
                                        wgpu::BufferBindingType::Uniform
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size: wgpu::BufferSize::new(light_uniform_size),
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    multisampled: false,
                                    sample_type: wgpu::TextureSampleType::Depth,
                                    view_dimension: wgpu::TextureViewDimension::D2Array,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(
                                    wgpu::SamplerBindingType::Comparison,
                                ),
                                count: None,
                            },
                        ],
                        label: None,
                    });
            let pipeline_layout =
                gpu.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("main"),
                        bind_group_layouts: &[&bind_group_layout, &local_bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let mx_total =
                Self::generate_matrix(gpu.config.width as f32 / gpu.config.height as f32);
            let forward_uniforms = GlobalUniforms {
                proj: mx_total.to_cols_array_2d(),
                num_lights: [lights.len() as u32, 0, 0, 0],
            };
            let uniform_buf = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Uniform Buffer"),
                    contents: bytemuck::bytes_of(&forward_uniforms),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

            // Create bind group
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: light_storage_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&shadow_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                    },
                ],
                label: None,
            });

            // Create the render pipeline
            let pipeline = gpu
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("main"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[vb_desc],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(if supports_storage_resources {
                            "fs_main"
                        } else {
                            "fs_main_without_storage"
                        }),
                        compilation_options: Default::default(),
                        targets: &[Some(gpu.config.format.into())],
                    }),
                    primitive: wgpu::PrimitiveState {
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: Self::DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });

            Pass {
                pipeline,
                bind_group,
                uniform_buf,
            }
        };

        let forward_depth = Self::create_depth_texture(&gpu.config, &gpu.device);

        Self {
            entities,
            lights,
            lights_are_dirty: true,
            shadow_pass,
            forward_pass,
            forward_depth,
            light_storage_buf,
            entity_uniform_buf,
            entity_bind_group,
        }
    }

    fn resize(&mut self, gpu: &mut Gpu) {
        // update view-projection matrix
        let mx_total = Self::generate_matrix(gpu.config.width as f32 / gpu.config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        gpu.queue.write_buffer(
            &self.forward_pass.uniform_buf,
            0,
            bytemuck::cast_slice(mx_ref),
        );

        self.forward_depth = Self::create_depth_texture(&gpu.config, &gpu.device);
    }

    fn render(&mut self, view: &wgpu::TextureView, gpu: &Gpu) {
        // update uniforms

        let entity_uniform_size = size_of::<EntityUniforms>() as wgpu::BufferAddress;
        let num_entities = self.entities.len() as wgpu::BufferAddress;
        // Make the `uniform_alignment` >= `entity_uniform_size` and aligned to `min_uniform_buffer_offset_alignment`.
        let uniform_alignment = {
            let alignment =
                gpu.device.limits().min_uniform_buffer_offset_alignment as wgpu::BufferAddress;
            wgpu::util::align_to(entity_uniform_size, alignment)
        };
        if self.entity_uniform_buf.size() < uniform_alignment * num_entities {
            let new_size = uniform_alignment * num_entities * 2;
            self.entity_uniform_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: new_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        for (i, entity) in self.entities.iter_mut().enumerate() {
            if entity.rotation_speed != 0.0 {
                let rotation = glam::Mat4::from_rotation_x(entity.rotation_speed * PI / 180.);
                entity.world *= rotation;
            }
            let data = EntityUniforms {
                model: entity.world.to_cols_array_2d(),
                color: [
                    entity.color.x,
                    entity.color.y,
                    entity.color.z,
                    entity.color.w,
                ],
            };
            gpu.queue.write_buffer(
                &self.entity_uniform_buf,
                i as wgpu::BufferAddress * uniform_alignment,
                bytemuck::bytes_of(&data),
            );
        }

        if self.lights_are_dirty {
            self.lights_are_dirty = false;
            for (i, light) in self.lights.iter().enumerate() {
                gpu.queue.write_buffer(
                    &self.light_storage_buf,
                    (i * size_of::<LightRaw>()) as wgpu::BufferAddress,
                    bytemuck::bytes_of(&light.to_raw()),
                );
            }
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.push_debug_group("shadow passes");
        for (i, light) in self.lights.iter().enumerate() {
            encoder.push_debug_group(&format!(
                "shadow pass {} (light at position {:?})",
                i, light.pos
            ));

            // The light uniform buffer already has the projection,
            // let's just copy it over to the shadow uniform buffer.
            encoder.copy_buffer_to_buffer(
                &self.light_storage_buf,
                (i * size_of::<LightRaw>()) as wgpu::BufferAddress,
                &self.shadow_pass.uniform_buf,
                0,
                64,
            );

            encoder.insert_debug_marker("render entities");
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &light.target_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.shadow_pass.pipeline);
                pass.set_bind_group(0, &self.shadow_pass.bind_group, &[]);

                for (i, entity) in self.entities.iter().enumerate() {
                    pass.set_bind_group(
                        1,
                        &self.entity_bind_group,
                        &[i as wgpu::DynamicOffset * uniform_alignment as wgpu::DynamicOffset],
                    );
                    entity.mesh.draw(&mut pass);
                }
            }

            encoder.pop_debug_group();
        }
        encoder.pop_debug_group();

        // forward pass
        encoder.push_debug_group("forward rendering pass");
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.forward_depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.forward_pass.pipeline);
            pass.set_bind_group(0, &self.forward_pass.bind_group, &[]);

            for (i, entity) in self.entities.iter().enumerate() {
                pass.set_bind_group(
                    1,
                    &self.entity_bind_group,
                    &[i as wgpu::DynamicOffset * uniform_alignment as wgpu::DynamicOffset],
                );
                entity.mesh.draw(&mut pass);
            }
        }
        encoder.pop_debug_group();

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    gpu: Gpu,
    is_surface_configured: bool,
    example: Example,
}

impl AppState {
    async fn new(
        event_loop: &ActiveEventLoop,
        window_attributes: WindowAttributes,
    ) -> anyhow::Result<Self> {
        let window = Arc::new(event_loop.create_window(window_attributes)?);
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: Example::FEATURES,
                required_limits: wgpu::Limits::defaults(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;
        let surface_capabilities = surface.get_capabilities(&adapter);
        let surface_format = surface_capabilities
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_capabilities.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_capabilities.present_modes[0],
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
        };

        let gpu = Gpu {
            device,
            queue,
            config,
        };

        let example = Example::new(&adapter, &gpu);

        Ok(Self {
            window,
            surface,
            gpu,
            is_surface_configured: false,
            example,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width != 0 && height != 0 {
            self.gpu.config.width = width;
            self.gpu.config.height = height;
            self.surface.configure(&self.gpu.device, &self.gpu.config);
            self.is_surface_configured = true;
            self.example.resize(&mut self.gpu);
        }
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {}
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.example.render(&view, &self.gpu);
        output.present();
        Ok(())
    }
}

struct App {
    app_state: Option<AppState>,
    result: anyhow::Result<()>,
}

impl App {
    fn new() -> Self {
        Self {
            app_state: None,
            result: Ok(()),
        }
    }

    fn create_app_state(&self, event_loop: &ActiveEventLoop) -> anyhow::Result<AppState> {
        let window_attributes = Window::default_attributes();
        pollster::block_on(AppState::new(event_loop, window_attributes))
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        match self.create_app_state(event_loop) {
            Ok(app_state) => self.app_state = Some(app_state),
            Err(error) => {
                self.result = Err(anyhow::anyhow!(
                    "Failed to create application state, error: {error}"
                ));
                event_loop.exit();
            }
        };
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(app_state) = self.app_state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => app_state.resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => match event.physical_key {
                PhysicalKey::Code(key_code) => {
                    app_state.handle_key(event_loop, key_code, event.state.is_pressed())
                }
                PhysicalKey::Unidentified(_native_key_code) => {}
            },
            WindowEvent::RedrawRequested => match app_state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    let size = app_state.window.inner_size();
                    app_state.resize(size.width, size.height);
                }
                Err(error) => log::error!("Unable to render {error}"),
            },
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    app.result
}
