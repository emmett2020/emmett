<template>
  <StudyItem></StudyItem>
  <!-- <WeatherItem></WeatherItem> -->
</template>

<script lang="ts">
import { defineComponent } from "vue";
// import WeatherItem from "./components/WeatherItem.vue";
import StudyItem from "./components/StudyItem.vue";

export default defineComponent({
  name: "App",
  components: {
    // HelloWorld,
    // WeatherItem,
    StudyItem,
  },
});
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
