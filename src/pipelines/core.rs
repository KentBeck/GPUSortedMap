use std::sync::Arc;
use wgpu::Device;

pub struct ComputeStep {
    device: Arc<Device>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputeStep {
    pub fn new(
        device: Arc<Device>,
        shader_source: &str,
        entry_point: &str,
        bind_group_layout_entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry_point),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{}-bgl", entry_point)),
            entries: bind_group_layout_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}-pl", entry_point)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}-pipeline", entry_point)),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            device,
            pipeline,
            bind_group_layout,
        }
    }

    pub fn create_bind_group(
        &self,
        label: &str,
        entries: &[wgpu::BindGroupEntry],
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bind_group_layout,
            entries,
        })
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass_label: &str,
        bind_group: &wgpu::BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pass_label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
}
