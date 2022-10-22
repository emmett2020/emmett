<template>
  <div id="main" position="absolute" width="100px" height="100px">
    <img src="../assets/clock.png" />
    <img src="../assets/clock.png" />
    <p>hello world</p>
    <!-- <img src="../assets/clock.png" top="10px" width="100px" height="100px" /> -->
  </div>
</template>

<script lang="ts">
import { ref, defineComponent } from "vue";

export default defineComponent({
  setup() {
    let name = "WeatherItem";
    return { name };
  },
});
</script>

<style scoped>
img {
  position: absolute;
  height: 80px;
  width: 80px;
}
p {
  font-size: 20px;
}
</style>
