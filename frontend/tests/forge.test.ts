// node --experimental-strip-types --test tests/forge.test.ts
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { buildGraph, hiresSize, parseLoras, sameGraph, type Settings } from '../src/forge/graph.ts'
import { fromParameters, readComfyGraph, unrunnable } from '../src/forge/png.ts'

const base: Settings = {
  checkpoint: 'juggernaut.safetensors', clipSkip: 2, prompt: 'a lighthouse, <lora:Watercolor:0.8>', negative: 'blurry', sampler: 'euler_ancestral',
  scheduler: 'normal', steps: 28, cfg: 4.5, width: 832, height: 1216, seed: 5, batchCount: 1, batchSize: 1, hires: false,
  hrUpscaler: '4x_foolhardy_Remacri.pth', hrSteps: 14, hrDenoise: 0.4, hrScale: 1.5, hrResizeX: 0, hrResizeY: 0, hrResize: 'lanczos',
  hrSampler: '', hrScheduler: '', hrCfg: 0, up: false, upModel: '4x_foolhardy_Remacri.pth', upScale: 1.5, upResizeX: 0, upResizeY: 0,
  upResize: 'lanczos', denoise: 0.6,
}
const loras = ['WatercolorV1.safetensors', 'Watercolor.safetensors', 'styles/Bonfire.safetensors']
const classes = (g: Record<string, { class_type: string }>) => Object.values(g).map((n) => n.class_type)
type G = Record<string, { class_type: string; inputs: Record<string, any> }>
/** a graph read into the form and built again: the settings, what was left out, and the graph */
function roundTrip(g: G, files = loras) {
  const r = readComfyGraph(JSON.stringify(g))
  const s = { ...base, ...r.settings } as Settings
  return { ...r, graph: buildGraph(s, s.seed, files, 'rustyui') }
}

test('lora tags come out of the prompt and match files by name', () => {
  const r = parseLoras('wolf, <lora:watercolor:0.7>, beach <lora:Bonfire>', loras)
  assert.equal(r.text, 'wolf, beach')
  assert.deepEqual(r.loras, [{ file: 'Watercolor.safetensors', model: 0.7, clip: 0.7 }, { file: 'styles/Bonfire.safetensors', model: 1, clip: 1 }])
  assert.throws(() => parseLoras('<lora:nobody:1>', loras), /no LoRA called nobody/)
})

test("lora tags take only their separator with them, and A1111's te:unet weights", () => {
  assert.equal(parseLoras('rich textures,soft light,, <lora:Watercolor:1>', loras).text, 'rich textures,soft light,')
  assert.equal(parseLoras('<lora:Watercolor:1>, <lora:Bonfire:0.5>, a wolf', loras).text, 'a wolf')
  assert.equal(parseLoras('<lora:Watercolor:1>', loras).text, '')
  assert.deepEqual(parseLoras('x <lora:Watercolor:1:0.85>', loras).loras, [{ file: 'Watercolor.safetensors', model: 0.85, clip: 1 }])
  assert.deepEqual(parseLoras('x <lora:Watercolor:te=0.5:unet=0.9>', loras).loras, [{ file: 'Watercolor.safetensors', model: 0.9, clip: 0.5 }])
  const g = buildGraph({ ...base, prompt: 'x, <lora:Watercolor:0:0.7>' }, 1, loras, 'rustyui')
  assert.deepEqual(classes(g).slice(0, 3), ['CheckpointLoaderSimple', 'LoraLoaderModelOnly', 'CLIPSetLastLayer'])
})

test('txt2img: checkpoint, LoRA, clip skip, prompts, sampler, decode, save', () => {
  const g = buildGraph(base, 5, loras, 'rustyui')
  assert.deepEqual(classes(g), ['CheckpointLoaderSimple', 'LoraLoader', 'CLIPSetLastLayer', 'CLIPTextEncode', 'CLIPTextEncode', 'EmptyLatentImage', 'KSampler', 'VAEDecode', 'SaveImage'])
  const ks = Object.values(g).find((n) => n.class_type === 'KSampler')!.inputs
  assert.equal(ks.seed, 5)
  assert.equal(ks.denoise, 1)
  assert.equal(Object.values(g).find((n) => n.class_type === 'CLIPSetLastLayer')!.inputs.stop_at_clip_layer, -2)
})

test('hires fix through a model upscaler and in latent space', () => {
  const g = buildGraph({ ...base, hires: true }, 5, loras, 'rustyui')
  const c = classes(g)
  assert.ok(c.includes('UpscaleModelLoader') && c.includes('ImageUpscaleWithModel') && c.filter((x) => x === 'KSampler').length === 2)
  const scale = Object.values(g).find((n) => n.class_type === 'ImageScale')!.inputs
  assert.deepEqual([scale.width, scale.height], [1248, 1824])
  const lat = buildGraph({ ...base, hires: true, hrUpscaler: 'Latent' }, 5, loras, 'rustyui')
  assert.equal(Object.values(lat).find((n) => n.class_type === 'LatentUpscale')!.inputs.upscale_method, 'bilinear')
  assert.deepEqual(hiresSize({ ...base, hrResizeX: 1024 }), [1024, 1496])
})

test('BREAK joins encoded parts end to end', () => {
  const g = buildGraph({ ...base, prompt: 'a BREAK b' }, 1, loras, 'rustyui')
  assert.equal(classes(g).filter((x) => x === 'ConditioningConcat').length, 1)
  assert.deepEqual(Object.values(g).filter((n) => n.class_type === 'CLIPTextEncode').map((n) => n.inputs.text), ['a', 'b', 'blurry'])
})

test('img2img: the picture resized, encoded and partly redrawn', () => {
  const g = buildGraph({ ...base, initImage: 'x.png', denoise: 0.55 }, 9, loras, 'rustyui')
  const c = classes(g)
  assert.ok(c.includes('LoadImage') && c.includes('VAEEncode') && !c.includes('EmptyLatentImage'))
  assert.equal(Object.values(g).find((n) => n.class_type === 'KSampler')!.inputs.denoise, 0.55)
})

test('A1111 parameters', () => {
  const { settings: s, leftOut } = fromParameters('a wolf, beach\nNegative prompt: lowres, blurry\nSteps: 30, Sampler: DPM++ 2M Karras, CFG scale: 5.5, Seed: 123, Size: 832x1216, Model: juggernaut, Clip skip: 2, Hires upscale: 1.5, Hires steps: 12, Hires upscaler: 4x_foolhardy_Remacri, Denoising strength: 0.35')
  assert.equal(s.prompt, 'a wolf, beach')
  assert.equal(s.negative, 'lowres, blurry')
  assert.equal(s.sampler, 'dpmpp_2m')
  assert.equal(s.scheduler, 'karras')
  assert.deepEqual([s.steps, s.cfg, s.seed, s.width, s.height, s.clipSkip], [30, 5.5, 123, 832, 1216, 2])
  assert.equal(s.hires, true)
  assert.equal(s.hrDenoise, 0.35)
  assert.deepEqual(leftOut, [])
})

test("Forge's hires keys, a postprocess upscale, and the keys the form has no place for", () => {
  const r = fromParameters('a fox\nNegative prompt: blurry\nSteps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 7, Seed: 9, Size: 832x1216, Model: juggernaut, Denoising strength: 0.4, Hires resize: 1280x1856, Hires steps: 10, Hires upscaler: 4x_foolhardy_Remacri, Hires sampler: DPM++ 2M, Hires schedule type: Use same scheduler, Hires CFG Scale: 5, Postprocess upscaler: 4x_foolhardy_Remacri, Postprocess upscale to: 1920x2808, ADetailer model: face_yolov8n.pt, Model hash: 1663b28c5a30, Version: f2.0.1')
  const s = r.settings
  assert.deepEqual([s.scheduler, s.hires, s.hrResizeX, s.hrResizeY, s.hrSampler, s.hrScheduler, s.hrCfg], ['karras', true, 1280, 1856, 'dpmpp_2m', '', 5])
  assert.deepEqual([s.up, s.upModel, s.upResizeX, s.upResizeY], [true, '4x_foolhardy_Remacri', 1920, 2808])
  assert.deepEqual(r.leftOut, ['ADetailer model: face_yolov8n.pt'])
})

test("the form's own graph read back and built again is the same graph", () => {
  for (const s of [{ ...base, hires: true }, { ...base, hires: true, hrUpscaler: 'Latent', hrResizeX: 1000, hrResizeY: 1400 }, { ...base, prompt: 'a, BREAK b <lora:Bonfire:0.6:1>' }]) {
    const g = buildGraph(s, 77, loras, 'rustyui')
    const r = roundTrip(g)
    assert.deepEqual(r.leftOut, [])
    assert.ok(sameGraph(g, r.graph))
  }
  const r = readComfyGraph(JSON.stringify(buildGraph({ ...base, hires: true }, 77, loras, 'x')))
  assert.equal(r.settings.prompt, 'a lighthouse, <lora:Watercolor:0.8>')
  assert.deepEqual([r.settings.seed, r.settings.width, r.settings.height, r.settings.clipSkip, r.settings.hrScale, r.settings.hrSteps], [77, 832, 1216, 2, 1.5, 14])
})

// the shapes of typical SDXL workflows, prompts aside
const ckpt = { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'sdxl_model.safetensors' } }
const lora = (name: string, m: number, c: number, from: string) => ({ class_type: 'LoraLoader', inputs: { lora_name: name, strength_model: m, strength_clip: c, model: [from, 0], clip: [from, 1] } })
const enc = (text: string, clip: [string, number]) => ({ class_type: 'CLIPTextEncode', inputs: { text, clip } })
const ks = (seed: number, steps: number, latent: string, denoise = 1, model = '3') => ({
  class_type: 'KSampler', inputs: { cfg: 4, denoise, latent_image: [latent, 0], model: [model, 0], negative: ['10', 0], positive: ['9', 0], sampler_name: 'euler_ancestral', scheduler: 'normal', seed, steps },
})

test("rustyui's BREAK prompt with LoRAs and clip skip comes back exactly, commas and all", () => {
  // a script that writes ids as strings in its own order, ConditioningConcat nested to the left
  const g: G = {
    '1': ckpt, '2': lora('Watercolor.safetensors', 1, 1, '1'), '3': lora('WatercolorV1.safetensors', 0.6, 0.6, '2'),
    '4': { class_type: 'CLIPSetLastLayer', inputs: { clip: ['3', 1], stop_at_clip_layer: -2 } },
    '5': enc('masterpiece,best quality,', ['4', 0]), '6': enc('a white dog,bedroom,', ['4', 0]),
    '7': { class_type: 'ConditioningConcat', inputs: { conditioning_from: ['6', 0], conditioning_to: ['5', 0] } }, '8': enc('soft light,', ['4', 0]),
    '9': { class_type: 'ConditioningConcat', inputs: { conditioning_from: ['8', 0], conditioning_to: ['7', 0] } }, '10': enc('lowres,blurry,', ['4', 0]),
    '11': { class_type: 'EmptyLatentImage', inputs: { batch_size: 1, height: 1792, width: 1024 } }, '12': ks(610421101, 30, '11'),
    '13': { class_type: 'VAEDecode', inputs: { samples: ['12', 0], vae: ['1', 2] } }, '14': { class_type: 'SaveImage', inputs: { filename_prefix: 'x', images: ['13', 0] } },
  }
  const r = roundTrip(g)
  assert.deepEqual(r.leftOut, [])
  assert.equal(r.settings.prompt, 'masterpiece,best quality, BREAK a white dog,bedroom, BREAK soft light,, <lora:Watercolor:1>, <lora:WatercolorV1:0.6>')
  assert.ok(sameGraph(g, r.graph))
})

test('a LoRA with its own text encoder strength, and one on the model only', () => {
  const g: G = {
    '1': ckpt, '2': lora('Watercolor.safetensors', 0.85, 1, '1'), '3': { class_type: 'LoraLoaderModelOnly', inputs: { lora_name: 'styles/Bonfire.safetensors', strength_model: 0.5, model: ['2', 0] } },
    '9': enc('a wolf', ['2', 1]), '10': enc('blurry', ['2', 1]), '11': { class_type: 'EmptyLatentImage', inputs: { batch_size: 1, height: 1216, width: 832 } },
    '12': ks(7, 28, '11'), '13': { class_type: 'VAEDecode', inputs: { samples: ['12', 0], vae: ['1', 2] } }, '14': { class_type: 'SaveImage', inputs: { filename_prefix: 'x', images: ['13', 0] } },
  }
  const r = roundTrip(g)
  assert.equal(r.settings.prompt, 'a wolf, <lora:Watercolor:1:0.85>, <lora:styles/Bonfire:0:0.5>')
  assert.equal(r.settings.clipSkip, 1)
  assert.deepEqual(r.leftOut, [])
  assert.ok(sameGraph(g, r.graph))
})

test("a Krea 2 render: the prompt comes back, the model's three loaders are named as left out", () => {
  const g: G = {
    '1': { class_type: 'UNETLoader', inputs: { unet_name: 'krea2Turbo_v10.safetensors', weight_dtype: 'default' } },
    '2': { class_type: 'CLIPLoader', inputs: { clip_name: 'qwen3vl_4b_fp8_scaled.safetensors', type: 'krea2', device: 'default' } },
    '3': { class_type: 'VAELoader', inputs: { vae_name: 'qwen_image_vae.safetensors' } },
    '5': enc('an amateur photo of a car', ['20', 1]), '6': enc('tattoo', ['20', 1]),
    '7': { class_type: 'EmptyLatentImage', inputs: { width: 1024, height: 1536, batch_size: 1 } },
    '8': { class_type: 'KSampler', inputs: { model: ['20', 0], positive: ['5', 0], negative: ['6', 0], latent_image: ['7', 0], seed: 2036904292, steps: 8, cfg: 1.0, sampler_name: 'euler', scheduler: 'normal', denoise: 1.0 } },
    '9': { class_type: 'VAEDecode', inputs: { samples: ['8', 0], vae: ['3', 0] } },
    '10': { class_type: 'SaveImage', inputs: { images: ['9', 0], filename_prefix: 'car' } },
    '20': { class_type: 'LoraLoader', inputs: { lora_name: 'Inkwash.safetensors', strength_model: 0.9, strength_clip: 0.9, model: ['1', 0], clip: ['2', 0] } },
  }
  const r = readComfyGraph(JSON.stringify(g))
  assert.equal(r.settings.checkpoint, undefined)
  assert.equal(r.settings.prompt, 'an amateur photo of a car, <lora:Inkwash:0.9>')
  assert.deepEqual([r.settings.steps, r.settings.cfg, r.settings.width, r.settings.height], [8, 1, 1024, 1536])
  assert.equal(r.leftOut.length, 3)
  assert.match(r.leftOut.join('\n'), /VAELoader[\s\S]*UNETLoader[\s\S]*CLIPLoader, type krea2/)
  const info: any = { UNETLoader: { input: { required: { unet_name: [['krea2Turbo_v10.safetensors']] } } }, VAELoader: { input: { required: { vae_name: [['other.safetensors']] } } } }
  for (const c of ['CLIPTextEncode', 'EmptyLatentImage', 'KSampler', 'VAEDecode', 'SaveImage', 'LoraLoader']) info[c] = { input: { required: {} } }
  assert.deepEqual(unrunnable(JSON.stringify(g), info), ['rustyui here has no CLIPLoader', `VAELoader's vae_name "qwen_image_vae.safetensors" isn't one this backend has`])
})

test("a hires workflow: hires resized with bilinear, then a final upscale, all carried", () => {
  const g: G = {
    '1': ckpt, '3': lora('Watercolor.safetensors', 0.8, 0.8, '1'), '9': enc('a wolf', ['3', 1]), '10': enc('blurry', ['3', 1]),
    '7': { class_type: 'EmptyLatentImage', inputs: { width: 832, height: 1216, batch_size: 1 } }, '8': ks(71031, 28, '7'),
    '11': { class_type: 'VAEDecode', inputs: { samples: ['8', 0], vae: ['1', 2] } }, '12': { class_type: 'UpscaleModelLoader', inputs: { model_name: '4x_foolhardy_Remacri.pth' } },
    '13': { class_type: 'ImageUpscaleWithModel', inputs: { upscale_model: ['12', 0], image: ['11', 0] } },
    '14': { class_type: 'ImageScale', inputs: { image: ['13', 0], upscale_method: 'bilinear', width: 1280, height: 1856, crop: 'disabled' } },
    '15': { class_type: 'VAEEncode', inputs: { pixels: ['14', 0], vae: ['1', 2] } }, '16': ks(71031, 14, '15', 0.45),
    '17': { class_type: 'VAEDecode', inputs: { samples: ['16', 0], vae: ['1', 2] } }, '18': { class_type: 'UpscaleModelLoader', inputs: { model_name: '4x_foolhardy_Remacri.pth' } },
    '19': { class_type: 'ImageUpscaleWithModel', inputs: { upscale_model: ['18', 0], image: ['17', 0] } },
    '20': { class_type: 'ImageScale', inputs: { image: ['19', 0], upscale_method: 'lanczos', width: 1920, height: 2808, crop: 'disabled' } },
    '21': { class_type: 'SaveImage', inputs: { images: ['20', 0], filename_prefix: 'hires' } },
  }
  const r = roundTrip(g)
  const s = r.settings
  assert.deepEqual([s.hires, s.hrSteps, s.hrDenoise, s.hrResizeX, s.hrResizeY, s.hrResize], [true, 14, 0.45, 1280, 1856, 'bilinear'])
  assert.deepEqual([s.up, s.upModel, s.upResize, s.upScale, s.upResizeX, s.upResizeY], [true, '4x_foolhardy_Remacri.pth', 'lanczos', undefined, 1920, 2808])
  assert.deepEqual(r.leftOut, [])
  assert.ok(sameGraph(g, r.graph))
  // a hires pass with its own CFG and sampler
  g['16'] = { ...g['16'], inputs: { ...g['16'].inputs, cfg: 6, sampler_name: 'dpmpp_2m' } }
  const h = roundTrip(g)
  assert.deepEqual([h.settings.hrCfg, h.settings.hrSampler, h.settings.hrScheduler], [6, 'dpmpp_2m', ''])
  assert.ok(sameGraph(g, h.graph))
})
