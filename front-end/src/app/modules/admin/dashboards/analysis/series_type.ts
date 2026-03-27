export class LineSeries{
    name: string;
    type: string = 'line';
    data: number[]

    constructor(name: string, data: number[]) {
        this.name = name;
        this.data = data;
    }
}

export class BarSeries{
    name: string;
    type: string = 'column';
    data: number[]

    constructor(name: string, data: number[]) {
        this.name = name;
        this.data = data;
    }
}