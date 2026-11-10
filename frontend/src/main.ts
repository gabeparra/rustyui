import { createApp } from 'vue'

// /simple/ is the Forge-style editor; /simple/#graph rustyui's first editor, the node graph
if (location.hash === '#graph') {
  Promise.all([import('@comfyorg/litegraph/style.css'), import('./style.css'), import('./App.vue')]).then(([, , m]) => createApp(m.default).mount('#app'))
} else {
  import('./forge/ForgeApp.vue').then((m) => createApp(m.default).mount('#app'))
}
