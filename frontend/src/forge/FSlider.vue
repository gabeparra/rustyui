<script setup lang="ts">
// Forge's slider: the label and a number box on one line, the range under them
defineProps<{ label: string; min: number; max: number; step: number; modelValue: number }>()
const emit = defineEmits<{ 'update:modelValue': [v: number] }>()

function set(v: string) {
  const n = Number(v)
  if (v !== '' && !Number.isNaN(n)) emit('update:modelValue', n)
}
</script>

<template>
  <div class="fslider">
    <div class="fslider-head">
      <span class="fslider-label">{{ label }}</span>
      <input class="fslider-num" type="number" :min="min" :max="max" :step="step" :value="modelValue" @change="set(($event.target as HTMLInputElement).value)" />
    </div>
    <input class="fslider-range" type="range" :min="min" :max="max" :step="step" :value="modelValue" @input="set(($event.target as HTMLInputElement).value)" />
  </div>
</template>
