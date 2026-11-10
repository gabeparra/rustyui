<script setup lang="ts">
// The queue: every prompt the backend has run (newest first), whoever queued it, with its
// pictures, its time, and the graph it ran.
import { isVideo, viewUrl, type HistoryItem } from '../api'

defineProps<{ items: HistoryItem[] }>()
const emit = defineEmits<{
  open: [id: string]
  forget: [id: string]
  view: [o: { filename: string; subfolder: string }]
}>()

const STATES: Record<string, string> = { queued: 'queued', running: 'rendering', success: 'done', error: 'failed', lost: 'lost' }

function ago(t: number): string {
  const s = Math.max(0, Date.now() / 1000 - t)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return new Date(t * 1000).toLocaleDateString()
}
</script>

<template>
  <aside class="panel">
    <div class="phead">
      <span>Queue and history</span>
      <span class="pcount">{{ items.length || '' }}</span>
    </div>
    <div class="hist">
      <div v-if="!items.length" class="ctitle">Nothing queued yet. Ctrl+Enter queues the graph.</div>
      <div v-for="it in items" :key="it.id" class="card" :class="it.status">
        <div class="chead">
          <span class="num">#{{ it.number ?? '?' }}</span>
          <span class="state">{{ STATES[it.status] ?? it.status }}</span>
          <span class="when">{{ ago(it.queued) }}</span>
          <span v-if="typeof it.seconds === 'number'" class="secs">{{ it.seconds.toFixed(1) }} s</span>
        </div>
        <div v-if="it.title" class="ctitle" :title="it.title">{{ it.title }}</div>
        <div v-if="it.error" class="cerr">{{ it.error }}</div>
        <div v-if="it.outputs?.length" class="thumbs">
          <component
            :is="isVideo(o) ? 'video' : 'img'"
            v-for="o in it.outputs"
            :key="o.subfolder + '/' + o.filename"
            :src="viewUrl(o)"
            muted
            preload="metadata"
            :title="o.filename"
            loading="lazy"
            alt=""
            @click="emit('view', o)"
          />
        </div>
        <div class="acts">
          <button title="open the graph this prompt was queued with" @click="emit('open', it.id)">Open graph</button>
          <button
            v-if="it.status !== 'queued' && it.status !== 'running'"
            title="take it off this list (the picture stays where it is)"
            @click="emit('forget', it.id)"
          >
            Forget
          </button>
        </div>
      </div>
    </div>
  </aside>
</template>
