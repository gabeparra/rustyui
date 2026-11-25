import { createApp } from 'vue'

// / is the Forge-style form; /#graph the node-graph editor
if (location.hash === '#graph') {
  Promise.all([import('@comfyorg/litegraph/style.css'), import('./style.css'), import('./App.vue')]).then(([, , m]) => createApp(m.default).mount('#app'))
} else {
  import('./forge/ForgeApp.vue').then((m) => createApp(m.default).mount('#app'))
}
