import * as React from "react";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
interface CircularProgressProps {
  value: number;
  renderLabel?: (progress: number) => number | string;
  size?: number;
  strokeWidth?: number;
  circleStrokeWidth?: number;
  progressStrokeWidth?: number;
  shape?: "square" | "round";
  className?: string;
  progressClassName?: string;
  labelClassName?: string;
  showLabel?: boolean;
}
const CircularProgress = ({
  value,
  renderLabel,
  className,
  progressClassName,
  labelClassName,
  showLabel = true,
  shape = "round",
  size = 100,
  strokeWidth,
  circleStrokeWidth = 10,
  progressStrokeWidth = 10,
}: CircularProgressProps) => {
  const radius = size / 2 - 10;
  const circumference = Math.ceil(3.14 * radius * 2);
  const percentage = Math.ceil(circumference * ((100 - value) / 100));
  const viewBox = `-${size * 0.125} -${size * 0.125} ${size * 1.25} ${
    size * 1.25
  }`;

  // Color logic based on percentage
  const getProgressColor = (value: number) => {
    if (value <= 30) return "stroke-green-500"; // Low percentage - green (safe)
    if (value <= 60) return "stroke-yellow-500"; // Medium percentage - yellow (warning)
    if (value <= 80) return "stroke-orange-500"; // High percentage - orange (caution)
    return "stroke-red-500"; // Very high percentage - red (danger)
  };

  const getTextColor = (value: number) => {
    if (value <= 30) return "text-green-500"; // Low percentage - green (safe)
    if (value <= 60) return "text-yellow-500"; // Medium percentage - yellow (warning)
    if (value <= 80) return "text-orange-500"; // High percentage - orange (caution)
    return "text-red-500"; // Very high percentage - red (danger)
  };

  const progressColor = getProgressColor(value);
  const textColor = getTextColor(value);

  return (
    <div className="relative">
      <svg
        width={size}
        height={size}
        viewBox={viewBox}
        version="1.1"
        xmlns="http://www.w3.org/2000/svg"
        style={{ transform: "rotate(-90deg)" }}
        className="relative"
      >
        {/* Base Circle */}
        <circle
          r={radius}
          cx={size / 2}
          cy={size / 2}
          fill="transparent"
          strokeWidth={strokeWidth ?? circleStrokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset="0"
          className={cn("stroke-primary/25", className)}
        />
        {/* Progress */}
        <circle
          r={radius}
          cx={size / 2}
          cy={size / 2}
          strokeWidth={strokeWidth ?? progressStrokeWidth}
          strokeLinecap={shape}
          strokeDashoffset={percentage}
          fill="transparent"
          strokeDasharray={circumference}
          className={cn(progressColor, progressClassName)}
        />
      </svg>
      {showLabel && (
        <div
          className={cn(
            "absolute inset-0 flex items-center justify-center text-lg font-semibold",
            textColor,
            labelClassName
          )}
        >
          {renderLabel ? renderLabel(value) : `${Number.parseInt(value.toString())}%`}
        </div>
      )}
    </div>
  );
};
export default function CircularProgressDemo(props: {
  value: number;
  size?: number;
  strokeWidth?: number;
  circleStrokeWidth?: number;
  progressStrokeWidth?: number;
  shape?: "square" | "round";
  className?: string;
  progressClassName?: string;
  labelClassName?: string;
  showLabel?: boolean;
}){
  return (
    <div className="max-w-xs mx-auto w-full flex flex-col items-center">
      <CircularProgress 
        value={props.value} 
        size={120} 
        strokeWidth={10}
        showLabel={props.showLabel ?? true}
        className={props.className}
        progressClassName={props.progressClassName}
        labelClassName={props.labelClassName}
        shape={props.shape}
      />
    </div>
  );
}
