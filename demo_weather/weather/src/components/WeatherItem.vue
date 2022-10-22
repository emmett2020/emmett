<template class="root">
  <div id="main" class="weather">
    <img id="mainWeatherIcon" src="../assets/sunny.png" class="mainWeather" />
    <div class="time">{{ currentTime }}</div>
    <div id="line" class="vertical-line"></div>
    <div class="list">
      <img class="divImg1" src="../assets/clock.png" />
      <p class="divP1">即将大雨</p>
      <img class="divImg2" src="../assets/hot.png" />
      <p class="divP2">18度</p>
      <img class="divImg3" src="../assets/代办事项.png" />
      <p class="divP3">6项todo</p>
      <img class="divImg4" src="../assets/Windmill.png" />
      <p class="divP4">微风</p>
    </div>
  </div>
</template>

<script lang="ts">
import { ref, defineComponent } from "vue";

export default defineComponent({
  setup() {
    let currentTime = ref("16:00");
    let name = "WeatherItem";
    return { currentTime, name };
  },
});
</script>

<style scoped>
.weather {
  position: relative;
  width: 500px;
  height: 316px;
  background-color: white;
  border-radius: 16px;
  filter: drop-shadow(0px 4px 4px rgba(0, 0, 0, 0.25));
}

.mainWeather {
  position: absolute;
  width: 96px;
  height: 96px;
  left: 14.6%;
  right: 66.2%;
  top: 23.73%;
  bottom: 45.89%;
}
.time {
  position: absolute;
  left: 6.8%;
  right: 61.2%;
  top: 57.28%;
  bottom: 26.27%;
  font-family: "Monaco";
  font-style: normal;
  font-weight: 400;
  font-size: 40px;
  line-height: 53px;
  text-align: center;
  color: #1d1b1b;
}

.vertical-line {
  position: absolute;
  width: 0px;
  height: 80%;
  left: 40%;
  top: 10%;
  border: 1px solid #eceeec;
}

.divImg,
.divImg1,
.divImg2,
.divImg3,
.divImg4 {
  position: absolute;
  width: 48px;
  height: 48px;
}

.divImg1 {
  width: 44px;
  height: 48px;
  top: 0%;
  left: 0%;
}
.divImg2 {
  width: 44px;
  height: 44px;
  top: 22%;
  left: 0%;
}
.divImg3 {
  width: 44px;
  height: 48px;
  top: 44%;
  left: 0%;
}
.divImg4 {
  width: 44px;
  height: 44px;
  top: 66%;
  left: 0%;
}
.divP,
.divP1,
.divP2,
.divP3,
.divP4 {
  position: absolute;
  width: 124px;
  height: 36px;
  font-family: "Monaco";
  font-style: normal;
  font-weight: 400;
  font-size: 18px;
  line-height: 29px;
  text-align: left;
  color: #000000;
}

.divP1 {
  top: -3%;
  left: 22%;
}
.divP2 {
  top: 18%;
  left: 22%;
}

.divP3 {
  top: 40%;
  left: 22%;
}
.divP4 {
  top: 62%;
  left: 22%;
}
.list {
  position: absolute;
  left: 45%;
  right: 70%;
  top: 18%;
  width: 300px;
  height: 250px;
}

.notify {
  position: absolute;
  left: 50%;
  top: -10%;
}
</style>
