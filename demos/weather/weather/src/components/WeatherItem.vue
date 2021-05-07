<template class="root">
  <div id="main" class="weather" draggable="true">
    <img id="mainWeatherIcon" src="../assets/sunny.png" class="mainWeather" />
    <div class="time">{{ currentTime }}</div>
    <div id="line" class="vertical-line"></div>
    <div class="list">
      <img
        class="divImg1"
        src="../assets/clock.png"
        @click="showWeatherDetail"
      />
      <p class="divP1">{{ notice }}</p>
      <img class="divImg2" src="../assets/hot.png" />
      <p class="divP2">{{ temperature }}</p>
      <img
        class="divImg3"
        src="../assets/代办事项.png"
        @click="showTodoDetail"
      />
      <p class="divP3">{{ todo }}</p>
      <img class="divImg4" src="../assets/Windmill.png" />
      <p class="divP4">{{ wind }}</p>
    </div>
    <div class="noticeExceptionWeatherStyle" v-if="showNotice">
      <img class="noticeExceptionWeatherImg" :src="noticeExceptWeatherImg" />
      <p class="noticeExceptionWeatherText">
        {{ noticeExceptWeatherData }}
      </p>
    </div>
    <div class="jumpToTodoStyle" v-if="showJumpToTodo">
      <a class="jumpToTodo">前往todo查看详情</a>
    </div>
  </div>
</template>

<script lang="ts">
import { ref, defineComponent, onMounted } from "vue";

export default defineComponent({
  components: {},
  setup(props) {
    let currentTime = ref("16:00");
    let wind = ref("微风");
    let temperature = ref("18度");
    let notice = ref("即将大雨");
    let name = "WeatherItem";
    let todo = ref("6项todo");
    let showNotice = ref(false);
    let showJumpToTodo = ref(false);
    function updateTime() {
      var data = new Date();
      var hours = "";
      var minutes = "";
      if (data.getHours() < 10) {
        hours += "0" + data.getHours();
      } else {
        hours = String(data.getHours());
      }
      if (data.getMinutes() < 10) {
        minutes += "0" + data.getMinutes();
      } else {
        minutes = String(data.getMinutes());
      }
      var t = hours + ":" + minutes;
      currentTime.value = String(t);
    }

    onMounted(() => {
      setInterval(updateTime, 1000);
    });

    let noticeExceptWeatherData = ref("08:00");
    let t = new URL("../assets/clock.png", import.meta.url).href;
    let noticeExceptWeatherImg = ref(t);
    const showWeatherDetail = () => {
      noticeExceptWeatherData.value = "08:01";
      noticeExceptWeatherImg.value = new URL(
        "../assets/drizzle.png",
        import.meta.url
      ).href;
      showNotice.value = !showNotice.value;
    };

    function showTodoDetail() {
      showJumpToTodo.value = !showJumpToTodo.value;
    }

    return {
      currentTime,
      name,
      wind,
      temperature,
      notice,
      todo,
      showNotice,
      showJumpToTodo,
      showWeatherDetail,
      showTodoDetail,
      noticeExceptWeatherData,
      noticeExceptWeatherImg,
    };
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
.divP4,
.noticeExceptionWeatherText,
.jumpToTodo {
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
  user-select: text;
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

.jumpToTodoStyle,
.noticeExceptionWeatherStyle {
  position: absolute;
  left: 55%;
  top: 20%;
  width: 98px;
  height: 120px;
  border: 1px;
  border-radius: 16px;
  background-color: #f38484;
  filter: drop-shadow(0px 4px 4px rgba(0, 0, 0, 0.25));
}

.noticeExceptionWeatherImg {
  position: absolute;
  width: 48px;
  height: 48px;
  top: 25%;
  left: 25%;
}

.noticeExceptionWeatherText {
  position: absolute;
  width: 50px;
  height: 30px;
  top: 50%;
  left: 25%;
  font-size: 16px;
}

.notify {
  position: absolute;
  left: 50%;
  top: -10%;
}

.jumpToTodoStyle {
  position: absolute;
  top: 55%;
}

.jumpToTodo {
  position: absolute;
  left: 0%;
  top: 25%;
  /* text-align: center; */
  text-justify: inter-word;
  font-size: 10px;
}
</style>
