export const addScriptSync = (path: string, id: string) => {
    if (document.getElementById(id)) {
        return false
    }
    const xhrObj = new XMLHttpRequest()
    xhrObj.open("GET", path, false)
    xhrObj.setRequestHeader(
        "Accept",
        "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01"
    )
    xhrObj.send("")
    const scriptElement = document.createElement("script")
    scriptElement.type = "text/javascript"
    scriptElement.text = xhrObj.responseText
    scriptElement.id = id
    document.head.appendChild(scriptElement)
}

// 本函数用于添加一个<script src=`$path` id=`${id}`><script>
export const addScript = (path: string, id: string) => {
    return new Promise((resolve, reject) => {
        // 创建script脚本前，该id正在被使用，返回添加失败即可
        if (document.getElementById(id)) {
            // 脚本加载后再次调用直接返回
            resolve() // 设置上future
            return false
        }
        const scriptElement = document.createElement("script")
        scriptElement.src = path
        scriptElement.async = true
        // 循环调用时 Chrome 不会重复请求 js
        document.head.appendChild(scriptElement)
        scriptElement.onload = () => {
            // 在创建的过程中， id存在了
            if (document.getElementById(id)) {
                // 循环调用需清除 DOM 中的 script 标签
                scriptElement.remove()
                resolve()
                return false
            }
            scriptElement.id = id
            resolve()
        }
    })
}
