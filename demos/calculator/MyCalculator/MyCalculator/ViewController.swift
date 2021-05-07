//
//  ViewController.swift
//  MyCalculator
//
//  Created by 张乃港 on 2022/3/13.
//

import UIKit

class ViewController: UIViewController {


    @IBOutlet weak var display: UILabel!
    
    
    var num1: Double?   //操作数1
    var num2: Double?   //操作数2
    var ope: String?    //运算符
    var IsFloat = false //浮点运算标志位
    var displayValue : Double {
        get{
            return Double(display.text!)!
        }
        set{
            display.text = String(newValue)
        }
    }

    
    @IBAction func touch(_ sender: UIButton) {
        
        let digit = sender.titleLabel?.text
    //获取按钮值
        let textInDisplay = display.text!  //获取标签值
        

        if let mathematicalSymbol = sender.titleLabel?.text {
            switch mathematicalSymbol {
                case "0","1","2","3","4","5","6","7","8","9":
                    if textInDisplay != "0" {
                        display.text = textInDisplay + digit!
                    } else {
                        display.text = digit
                    }
                case "AC": display.text = "0"
                case "π": display.text = String(Double.pi)
                case "√": display.text = String(sqrt(displayValue))
                case "sin": display.text = String(sin(displayValue))
                case "+","-","*","/":
                    if textInDisplay.contains(".") {
                        IsFloat = true
                    }
                    self.ope = mathematicalSymbol   //记录运算符
                    self.num1 = displayValue        //记录操作数1
                    display.text = "0"              //清空计算器显示
                case "=":
                    if textInDisplay.contains(".") {
                        IsFloat = true
                    }
                    self.num2 = displayValue
                    display.text = calculation(num1: num1, num2: num2, ope: ope, IsFloat: IsFloat)
                    IsFloat = false
                    case ".":
                        if textInDisplay != "0" {
                            display.text = textInDisplay + digit!
                        } else {
                            display.text = "0."
                        }
                    default:
                        break
            }
        }
    }
    
    
    //calculation：执行计算操作
    func calculation(num1: Double?, num2: Double?, ope: String?, IsFloat: Bool) -> String {
        if ope == nil {
            return "EOF"
        }
        if num1 == nil || num2 == nil {
            return "0"
        }
        var result: Double
        switch ope {
            case "+": result = num1! + num2!
            case "-": result = num1! - num2!
            case "*": result = num1! * num2!
            case "/":
                if(num1 != 0) {
                    result = num1! / num2!
                }
                else {
                    return "EOF"
                }
            default:
                return "EOF"
        }
        if IsFloat {
            return String(Float(result))
        } else {
            return String(Int(result))
        }
    }
    
}

